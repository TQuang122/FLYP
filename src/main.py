from ast import arg
import os
import numpy as np
import torch
from src.models.eval import evaluate
from src.models.flyp_loss import flyp_loss
from src.models.ce_ablation import ce_ablation
from src.models.modeling import ClassificationHead, CLIPEncoder, ImageClassifier
from src.models.utils import fisher_load
from src.args import parse_arguments
import logging
import random


def start_wandb(args, logging_path):
    if not os.environ.get('WANDB_API_KEY'):
        return None

    try:
        import wandb
    except ImportError:
        return None

    run_name = "_BS" + str(args.batch_size) + "_WD" + str(
        args.wd) + "_LR" + str(args.lr) + "_run" + str(args.run)
    return wandb.init(project=os.environ.get('WANDB_PROJECT', 'flyp-iwildcam'),
                      name=os.environ.get('WANDB_RUN_NAME', run_name),
                      group=os.environ.get('WANDB_GROUP', args.exp_name),
                      dir=logging_path,
                      config=vars(args))


def main(args):

    ###logging##################################################################
    os.makedirs(args.save + args.exp_name, exist_ok=True)
    args.save = args.save + args.exp_name + "/" + "_BS" + str(
        args.batch_size) + "_WD" + str(args.wd) + "_LR" + str(args.lr) + "_run" + str(args.run)
    os.makedirs("expt_logs/" + args.exp_name, exist_ok=True)
    logging_path = "expt_logs/" + args.exp_name + "/" + "_BS" + str(
        args.batch_size) + "_WD" + str(args.wd) + "_LR" + str(args.lr) + "_run" + str(args.run)
    os.makedirs(logging_path, exist_ok=True)
    log_filename = logging_path + "/log.log"
    logging.basicConfig(filename=log_filename,
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    assert args.save is not None, 'Please provide a path to store models'
    wandb_run = start_wandb(args, logging_path)
    args.use_wandb = wandb_run is not None
    #############################################################################

    # Initialize the CLIP encoder
    clip_encoder = CLIPEncoder(args, keep_lang=True)
    classification_head = ClassificationHead(normalize=True, weights=None)
    logger.info(args)
    if args.ce_ablation:
        finetuned_checkpoint = ce_ablation(args, clip_encoder,
                                            classification_head, logger)
    else:
        finetuned_checkpoint = flyp_loss(args, clip_encoder,
                                            classification_head, logger)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
