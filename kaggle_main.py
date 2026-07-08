import os
import subprocess
import sys
from pathlib import Path


KAGGLE_WORKING = Path("/kaggle/working")
SCRIPT_DIR = Path(__file__).resolve().parent
KAGGLE_SRC = Path("/kaggle/src")
KAGGLE_INPUT = Path("/kaggle/input")
SOURCE_REPO = os.environ.get("FLYP_SOURCE_REPO", "https://github.com/TQuang122/FLYP.git")


def is_source_root(path: Path) -> bool:
    return (path / "datacreation_scripts" / "iwildcam.py").is_file() and (
        path / "src" / "main.py"
    ).is_file()


def clone_source_root() -> Path | None:
    if not KAGGLE_WORKING.exists():
        return None

    for target in [KAGGLE_WORKING / "FLYP", KAGGLE_WORKING / "FLYP_source"]:
        if is_source_root(target):
            return target
        if target.exists():
            continue

        command = ["git", "clone", "--depth", "1", SOURCE_REPO, str(target)]
        print("\n$ " + " ".join(command), flush=True)
        subprocess.run(command, check=True)
        if is_source_root(target):
            return target

    return None


def find_source_root() -> Path:
    candidates = [SCRIPT_DIR, Path.cwd(), KAGGLE_SRC, KAGGLE_WORKING / "FLYP"]
    if KAGGLE_SRC.exists():
        candidates.extend(path.parent for path in KAGGLE_SRC.rglob("datacreation_scripts/iwildcam.py"))
    if KAGGLE_WORKING.exists():
        candidates.extend(path.parent for path in KAGGLE_WORKING.rglob("datacreation_scripts/iwildcam.py"))

    for candidate in candidates:
        if is_source_root(candidate):
            return candidate

    cloned_root = clone_source_root()
    if cloned_root is not None:
        return cloned_root

    checked = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find FLYP source tree containing datacreation_scripts/iwildcam.py "
        f"and src/main.py. Checked:\n{checked}"
    )


ROOT = find_source_root()
WORK_ROOT = KAGGLE_WORKING if KAGGLE_WORKING.exists() else ROOT
DATA_ROOT = WORK_ROOT / "datasets" / "data"
IWILDCAM_LINK = DATA_ROOT / "iwildcam_v2.0"
CSV_PATH = WORK_ROOT / "datasets" / "csv" / "iwildcam_v2.0" / "train.csv"
DEFAULT_KAGGLE_SOURCE = Path(
    "/kaggle/input/iwildcam-v2-0-2020-wilds-dataset/iwildcam_v2.0"
)


def run(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path = ROOT,
) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def install_dependencies() -> None:
    if os.environ.get("SKIP_INSTALL") == "1":
        print("Skipping dependency install because SKIP_INSTALL=1", flush=True)
        return

    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "open_clip_torch",
            "wilds",
            "braceexpand",
            "webdataset",
            "h5py",
            "pandas",
            "pillow",
            "tqdm",
            "scikit-learn",
            "wandb",
            "git+https://github.com/modestyachts/ImageNetV2_pytorch",
        ]
    )
    run([sys.executable, "-m", "pip", "cache", "purge"])


_WANDB_API_KEY = "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"  # <-- paste your WANDB_API_KEY here if Kaggle Secrets fails


def configure_wandb() -> None:
    if os.environ.get("WANDB_API_KEY"):
        return

    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        if _WANDB_API_KEY:
            os.environ["WANDB_API_KEY"] = _WANDB_API_KEY
            return
        print("W&B disabled: WANDB_API_KEY is not set", flush=True)
        return

    try:
        api_key = UserSecretsClient().get_secret("WANDB_API_KEY")
    except Exception as exc:
        print(f"W&B disabled: could not read Kaggle Secret WANDB_API_KEY ({exc})", flush=True)
        if _WANDB_API_KEY:
            os.environ["WANDB_API_KEY"] = _WANDB_API_KEY
            return
        return

    os.environ["WANDB_API_KEY"] = api_key


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if new in text:
        return
    if old not in text:
        raise ValueError(f"Could not patch expected text in {path}")
    path.write_text(text.replace(old, new, 1))


def enable_wandb_logging(source_root: Path) -> None:
    main_path = source_root / "src" / "main.py"

    replace_once(
        main_path,
        "import random\n\ndef main(args):",
        "import random\n\n\ndef start_wandb(args, logging_path):\n"
        "    if not os.environ.get('WANDB_API_KEY'):\n"
        "        return None\n\n"
        "    try:\n"
        "        import wandb\n"
        "    except ImportError:\n"
        "        return None\n\n"
        "    run_name = \"_BS\" + str(args.batch_size) + \"_WD\" + str(\n"
        "        args.wd) + \"_LR\" + str(args.lr) + \"_run\" + str(args.run)\n"
        "    return wandb.init(project=os.environ.get('WANDB_PROJECT', 'flyp-iwildcam'),\n"
        "                      name=os.environ.get('WANDB_RUN_NAME', run_name),\n"
        "                      group=os.environ.get('WANDB_GROUP', args.exp_name),\n"
        "                      dir=logging_path,\n"
        "                      config=vars(args))\n\n\ndef main(args):",
    )
    replace_once(
        main_path,
        "    assert args.save is not None, 'Please provide a path to store models'\n",
        "    assert args.save is not None, 'Please provide a path to store models'\n"
        "    wandb_run = start_wandb(args, logging_path)\n"
        "    args.use_wandb = wandb_run is not None\n",
    )
    replace_once(
        main_path,
        "    else:\n        finetuned_checkpoint = flyp_loss(args, clip_encoder,\n                                            classification_head, logger)\n",
        "    else:\n        finetuned_checkpoint = flyp_loss(args, clip_encoder,\n                                            classification_head, logger)\n\n"
        "    if wandb_run is not None:\n"
        "        wandb_run.finish()\n",
    )

def resolve_iwildcam_source() -> Path:
    env_source = os.environ.get("IWILDCAM_SOURCE")
    candidates = [Path(env_source)] if env_source else []
    candidates.extend([IWILDCAM_LINK, DEFAULT_KAGGLE_SOURCE])
    candidates.extend(discover_iwildcam_candidates())

    for candidate in candidates:
        if (candidate / "metadata.csv").is_file() and (candidate / "train").is_dir():
            return candidate

    checked = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find iWildCam data. Set IWILDCAM_SOURCE to the folder "
        "containing metadata.csv and train/. "
        f"Kaggle input mounts:\n{describe_kaggle_input()}\nChecked:\n{checked}"
    )


def discover_iwildcam_candidates() -> list[Path]:
    if not KAGGLE_INPUT.exists():
        return []

    candidates = []
    for metadata_path in KAGGLE_INPUT.rglob("metadata.csv"):
        dataset_dir = metadata_path.parent
        if (dataset_dir / "train").is_dir():
            candidates.append(dataset_dir)
    return candidates


def describe_kaggle_input() -> str:
    if not KAGGLE_INPUT.exists():
        return f"{KAGGLE_INPUT} does not exist"

    entries = sorted(path.name for path in KAGGLE_INPUT.iterdir())
    if not entries:
        return f"{KAGGLE_INPUT} is empty"
    return "\n".join(f"- {name}" for name in entries)


def link_dataset(source: Path) -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if IWILDCAM_LINK.is_symlink() or IWILDCAM_LINK.exists():
        if IWILDCAM_LINK.resolve() == source.resolve():
            print(f"Using existing dataset link: {IWILDCAM_LINK} -> {source}", flush=True)
            return
        if IWILDCAM_LINK.is_dir() and not IWILDCAM_LINK.is_symlink():
            raise FileExistsError(
                f"{IWILDCAM_LINK} already exists and is not a symlink. "
                "Remove it or set IWILDCAM_SOURCE to match it."
            )
        IWILDCAM_LINK.unlink()

    IWILDCAM_LINK.symlink_to(source, target_is_directory=True)
    print(f"Linked dataset: {IWILDCAM_LINK} -> {source}", flush=True)


def generate_csv() -> None:
    run(
        [
            sys.executable,
            str(ROOT / "datacreation_scripts" / "iwildcam.py"),
            "--save_file",
            str(CSV_PATH),
            "--metadata",
            str(IWILDCAM_LINK / "metadata.csv"),
            "--english_label_path",
            str(ROOT / "src" / "datasets" / "iwildcam_metadata" / "labels.csv"),
            "--data_dir",
            str(IWILDCAM_LINK / "train"),
        ],
        cwd=WORK_ROOT,
    )
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"CSV was not generated: {CSV_PATH}")
    print(f"CSV ready: {CSV_PATH}", flush=True)


def build_train_command() -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "src" / "main.py"),
        "--train-dataset=IWildCamIDVal",
        f"--epochs={os.environ.get('EPOCHS', '20')}",
        f"--lr={os.environ.get('LR', '1e-5')}",
        f"--wd={os.environ.get('WD', '0.2')}",
        f"--batch-size={os.environ.get('BATCH_SIZE', '256')}",
        f"--model={os.environ.get('MODEL', 'ViT-B/16')}",
        "--eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD",
        "--template=iwildcam_template",
        f"--save={os.environ.get('SAVE_DIR', str(WORK_ROOT / 'checkpoints'))}",
        f"--data-location={DATA_ROOT}",
        f"--ft_data={CSV_PATH}",
        "--csv-img-key=filepath",
        "--csv-caption-key=title",
        f"--exp_name={os.environ.get('EXP_NAME', 'iwildcam/flyp_loss')}",
        f"--workers={os.environ.get('WORKERS', '4')}",
        f"--run={os.environ.get('RUN_ID', '1')}",
    ]

    resume_path = os.environ.get("RESUME")
    if resume_path:
        cmd.append(f"--resume={resume_path}")

    keep_cp = os.environ.get("KEEP_CHECKPOINTS")
    if keep_cp:
        cmd.append(f"--keep-checkpoints={keep_cp}")

    microbatch = os.environ.get("MICROBATCH_SIZE")
    if microbatch:
        cmd.append(f"--microbatch-size={microbatch}")

    return cmd


def main() -> None:
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("MICROBATCH_SIZE", "32")
    os.environ.setdefault("WORKERS", "4")
    os.environ["PYTHONPATH"] = f"{ROOT}:{os.environ.get('PYTHONPATH', '')}"
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    (WORK_ROOT / "checkpoints").mkdir(exist_ok=True)
    (WORK_ROOT / "expt_logs").mkdir(exist_ok=True)

    install_dependencies()
    configure_wandb()
    enable_wandb_logging(ROOT)
    source = resolve_iwildcam_source()
    link_dataset(source)
    generate_csv()

    train_command = build_train_command()
    print("\nResolved training command:", flush=True)
    print(" ".join(train_command), flush=True)

    if os.environ.get("DRY_RUN") == "1":
        print("DRY_RUN=1, not starting training", flush=True)
        return

    run(train_command, env=os.environ.copy(), cwd=WORK_ROOT)

    stats_dir = WORK_ROOT / "expt_logs" / os.environ.get("EXP_NAME", "iwildcam/flyp_loss")
    print(f"\nTraining finished. Logs and stats are under: {stats_dir}", flush=True)
    print(f"Checkpoints are under: {WORK_ROOT / 'checkpoints'}", flush=True)


if __name__ == "__main__":
    main()
