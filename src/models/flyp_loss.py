from asyncio.constants import LOG_THRESHOLD_FOR_CONNLOST_WRITES
import os
import copy
import time
import tqdm

import torch
import pandas as pd
import clip.clip as clip
from clip.loss import ClipLoss

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models.eval import evaluate
from src.models.modeling import ClassificationHead, CLIPEncoder, ImageClassifier
from src.models.utils import cosine_lr, torch_load, LabelSmoothing, get_logits
from src.models.zeroshot import get_zeroshot_classifier
from src.datasets.laion import get_data
import src.datasets as datasets


def cleanup_old_checkpoints(save_dir, keep_last=3, logger=None):
    """Delete old checkpoint files, keeping only the `keep_last` most recent epochs."""
    if not os.path.isdir(save_dir):
        return
    import re as re_m
    ckpt_files = [f for f in os.listdir(save_dir) if re_m.match(r'checkpoint_\d+\.pt', f)]
    if keep_last > 0 and len(ckpt_files) <= keep_last:
        return
    epochs = sorted([int(re_m.search(r'checkpoint_(\d+)\.pt', f).group(1)) for f in ckpt_files])
    epochs_to_delete = epochs if keep_last <= 0 else epochs[:-keep_last]
    for ep in epochs_to_delete:
        for suffix in ['checkpoint', 'optim', 'scaler']:
            fpath = os.path.join(save_dir, f'{suffix}_{ep}.pt')
            if os.path.exists(fpath):
                os.remove(fpath)
                if logger:
                    logger.info(f'Deleted old checkpoint: {fpath}')
    if logger:
        logger.info(f'Cleaned up {len(epochs_to_delete)} old checkpoint(s), keeping last {keep_last}')


def prepare_checkpoint_dir_for_save(save_dir, keep_last=3, logger=None):
    cleanup_old_checkpoints(save_dir, keep_last=max(keep_last - 1, 0), logger=logger)


def log_wandb_epoch(args, epoch_stats):
    if not getattr(args, 'use_wandb', False):
        return

    import wandb

    metrics = {}
    for key, value in epoch_stats.items():
        metric_name = key.replace(' ', '/')
        metrics[metric_name] = value
    wandb.log(metrics, step=epoch_stats['epoch'])


def flyp_loss(args, clip_encoder, classification_head, logger):
    assert args.train_dataset is not None, "Please provide a training dataset."
    logger.info('Fine-tuning Using FLYP Loss')
    model = clip_encoder
    input_key = 'images'
    preprocess_fn = clip_encoder.train_preprocess
    image_enc = None
    clip_encoder.process_images = True
    print_every = 100

    dataset_class = getattr(datasets, args.train_dataset)
    print(f"Training dataset {args.train_dataset}")

    dataset = dataset_class(preprocess_fn,
                            location=args.data_location,
                            batch_size=args.batch_size)

    img_text_data = get_data(
        args, (clip_encoder.train_preprocess, clip_encoder.val_preprocess),
        epoch=0)
    assert len(
        img_text_data), 'At least one train or eval dataset must be specified.'
    ft_dataloader = img_text_data['train_ft'].dataloader
    ft_iterator = iter(ft_dataloader)
    num_batches = len(dataset.train_loader)
    print(f"Num batches is {num_batches}")

    model = model.cuda()
    classification_head = classification_head.cuda()
    devices = list(range(torch.cuda.device_count()))
    logger.info('Using devices' + str(devices))
    model = torch.nn.DataParallel(model, device_ids=devices)
    classification_head = torch.nn.DataParallel(classification_head,
                                                device_ids=devices)
    classification_head.train()
    model.train()

    clip_loss_fn = ClipLoss(local_loss=False,
                            gather_with_grad=False,
                            cache_labels=True,
                            rank=0,
                            world_size=1,
                            use_horovod=False)

    clip_params = list(model.parameters())
    total_params = clip_params
    params = [p for p in total_params if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    use_amp = os.environ.get('USE_AMP', '1') == '1'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length,
                          args.epochs * num_batches, args.min_lr)

    start_epoch = 0
    if args.resume:
        ckpt_dir = args.resume
        if os.path.isdir(ckpt_dir):
            import re as re_m
            ckpt_files = [f for f in os.listdir(ckpt_dir) if re_m.match(r'checkpoint_\d+\.pt', f)]
            if ckpt_files:
                epochs_found = sorted([int(re_m.search(r'checkpoint_(\d+)\.pt', f).group(1)) for f in ckpt_files])
                latest_epoch = epochs_found[-1]
                start_epoch = latest_epoch + 1

                ckpt_path = os.path.join(ckpt_dir, f'checkpoint_{latest_epoch}.pt')
                logger.info(f'Resuming from checkpoint: {ckpt_path}')
                logger.info(f'Starting from epoch {start_epoch} (resumed at epoch {latest_epoch})')

                model.module = model.module.load(ckpt_path)
                model = model.cuda()

                optim_path = os.path.join(ckpt_dir, f'optim_{latest_epoch}.pt')
                if not os.path.exists(optim_path):
                    optim_path = os.path.join(ckpt_dir, 'optim_latest.pt')
                if os.path.exists(optim_path):
                    try:
                        optim_state = torch.load(optim_path, map_location='cuda')
                        optimizer.load_state_dict(optim_state)
                        logger.info(f'Loaded optimizer state from {optim_path}')
                    except Exception as e:
                        logger.warning(f'Failed to load optimizer state from {optim_path}: {e}')
                        logger.info('Continuing without optimizer state (will start from scratch lr schedule)')

                scaler_path = os.path.join(ckpt_dir, f'scaler_{latest_epoch}.pt')
                if not os.path.exists(scaler_path):
                    scaler_path = os.path.join(ckpt_dir, 'scaler_latest.pt')
                if os.path.exists(scaler_path):
                    try:
                        scaler_state = torch.load(scaler_path, map_location='cuda')
                        scaler.load_state_dict(scaler_state)
                        logger.info(f'Loaded GradScaler state from {scaler_path}')
                    except Exception as e:
                        logger.warning(f'Failed to load GradScaler state from {scaler_path}: {e}')
            else:
                logger.warning(f'No checkpoints found in {ckpt_dir}, starting from scratch')
        else:
            logger.warning(f'Resume directory not found: {ckpt_dir}, starting from scratch')

    if start_epoch >= args.epochs:
        logger.info(f'start_epoch ({start_epoch}) >= args.epochs ({args.epochs}), nothing to train')
        return

    microbatch_size = getattr(args, 'microbatch_size', 0)
    if microbatch_size <= 0 or microbatch_size >= args.batch_size:
        try:
            microbatch_size = int(os.environ.get('MICROBATCH_SIZE', '0'))
        except (ValueError, TypeError):
            microbatch_size = 0

    use_microbatch = microbatch_size > 0 and microbatch_size < args.batch_size
    if use_microbatch:
        num_micro = (args.batch_size + microbatch_size - 1) // microbatch_size
        logger.info(f'Using gradient accumulation: {num_micro} microbatches of size {microbatch_size} '
                    f'(effective batch size {args.batch_size})')

    stats = []
    best_ood_acc = -1.0
    for epoch in range(start_epoch, args.epochs):
        print("Epoch : ", epoch)
        epoch_stats = {}
        epoch_stats['epoch'] = epoch
        id_flyp_loss_sum = 0
        model.train()
        model = model.cuda()
        classification_head.train()

        for i in range(num_batches):
            start_time = time.time()
            step = i + epoch * num_batches
            if epoch != -1:
                scheduler(step)
            optimizer.zero_grad(set_to_none=True)

            try:
                ft_batch = next(ft_iterator)
            except StopIteration:
                ft_iterator = iter(ft_dataloader)
                ft_batch = next(ft_iterator)

            ft_image, ft_text = ft_batch
            ft_image, ft_text = ft_image.cuda(), ft_text.cuda()

            if use_microbatch:
                batch_loss_val = 0.0
                for j in range(num_micro):
                    start_idx = j * microbatch_size
                    end_idx = min(start_idx + microbatch_size, args.batch_size)
                    micro_image = ft_image[start_idx:end_idx]
                    micro_text = ft_text[start_idx:end_idx]

                    with torch.cuda.amp.autocast(enabled=use_amp):
                        ft_image_features, ft_text_features, logit_scale2 = model(
                            micro_image, micro_text)
                        micro_loss = clip_loss_fn(ft_image_features,
                                                  ft_text_features,
                                                  logit_scale2[0])

                    scaler.scale(micro_loss / num_micro).backward()
                    batch_loss_val += micro_loss.item()

                scaler.step(optimizer)
                scaler.update()
                loss_val = batch_loss_val / num_micro
            else:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    ft_image_features, ft_text_features, logit_scale2 = model(
                        ft_image, ft_text)
                    ft_clip_loss = clip_loss_fn(ft_image_features,
                                                ft_text_features,
                                                logit_scale2[0])

                scaler.scale(ft_clip_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_val = ft_clip_loss.item()

            id_flyp_loss_sum += loss_val

            if i % print_every == 0:
                percent_complete = 100 * i / num_batches
                logger.info(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{num_batches}]\t"
                    f"ID FLYP Loss: {loss_val:.4f}")

        id_flyp_loss_avg = id_flyp_loss_sum / num_batches

        # Evaluate
        args.current_epoch = epoch
        classification_head_new = get_zeroshot_classifier(
            args, model.module.model)
        classification_head_new = classification_head_new.cuda()

        eval_results = evaluate(model, args, classification_head_new,
                                epoch_stats, logger)

        # Saving model
        if args.save is not None:
            os.makedirs(args.save, exist_ok=True)
            keep = getattr(args, 'keep_checkpoints', 3)
            prepare_checkpoint_dir_for_save(args.save, keep_last=keep, logger=logger)

            model_path = os.path.join(args.save, f'checkpoint_{epoch}.pt')
            logger.info('Saving model to' + str(model_path))
            model.module.save(model_path)
            checkpoint_state = getattr(args, 'checkpoint_state', 'latest')
            if checkpoint_state == 'all':
                optim_path = os.path.join(args.save, f'optim_{epoch}.pt')
                torch.save(optimizer.state_dict(), optim_path)
                scaler_path = os.path.join(args.save, f'scaler_{epoch}.pt')
                torch.save(scaler.state_dict(), scaler_path)
            elif checkpoint_state == 'latest':
                optim_path = os.path.join(args.save, 'optim_latest.pt')
                torch.save(optimizer.state_dict(), optim_path)
                scaler_path = os.path.join(args.save, 'scaler_latest.pt')
                torch.save(scaler.state_dict(), scaler_path)
            cleanup_old_checkpoints(args.save, keep_last=keep, logger=logger)

        ood_acc = 0
        num_datasets = 0
        for k, v in epoch_stats.items():
            if 'Accuracy' in k:
                if k == 'ImageNet Accuracy':
                    #ignore the ID acc term
                    continue
                ood_acc += v
                num_datasets += 1
        if num_datasets != 0:
            ood_acc = ood_acc / num_datasets
        else:
            ood_acc = 0

        epoch_stats['Avg OOD Acc'] = round(ood_acc, 4)
        logger.info(f"Avg OOD Acc : {ood_acc:.4f}")
        logger.info(f"Avg ID FLYP Loss : {id_flyp_loss_avg:.4f}")
        epoch_stats['Avg ID FLYP Loss'] = round(id_flyp_loss_avg, 4)

        if args.save is not None and ood_acc > best_ood_acc:
            best_ood_acc = ood_acc
            best_path = os.path.join(args.save, 'best.pt')
            logger.info(f'New best OOD acc {ood_acc:.4f}, saving to {best_path}')
            model.module.save(best_path)
            if getattr(args, 'checkpoint_state', 'latest') == 'all':
                best_optim_path = os.path.join(args.save, 'best_optim.pt')
                torch.save(optimizer.state_dict(), best_optim_path)
                best_scaler_path = os.path.join(args.save, 'best_scaler.pt')
                torch.save(scaler.state_dict(), best_scaler_path)
        stats.append(epoch_stats)
        stats_df = pd.DataFrame(stats)
        log_dir = "expt_logs/" + args.exp_name + "/" + "_BS" + str(
            args.batch_size) + "_WD" + str(args.wd) + "_LR" + str(args.lr) + "_run" + str(args.run)
        os.makedirs(log_dir, exist_ok=True)
        stats_df.to_csv(log_dir + '/stats.tsv', sep='\t')
        log_wandb_epoch(args, epoch_stats)

    if args.save is not None:
        return model_path
