import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
import torch.nn.functional as F
from torch import nn

from model import TableConditionalDiffusion
from train import build_dataframes, build_dataloaders, build_tokenizer, resolve_split_mode, setup_seed
from utils import evaluate_f1_stats, make_run_dir, save_json

CHECKPOINT_CONFIG_KEYS = (
    "shortcut_name",
    "max_length",
    "max_cols",
    "adaptive_col_budget",
    "y_dim",
    "cond_dim",
    "timesteps",
    "beta_start",
    "beta_end",
    "time_dim",
    "diffuser_type",
    "hidden_dim",
    "cond_drop_prob",
    "beta_schedule",
    "split_mode",
    "cv_fold",
    "valid_ratio",
    "test_ratio",
    "batch_size",
)

DEFAULT_EVAL_CONFIG = {
    "split_mode": "auto",
    "cv_fold": -1,
    "valid_ratio": 0.1,
    "test_ratio": 0.1,
    "batch_size": 16,
    "fast_steps": 100,
    "guidance_scale": 2.0,
    "mc_eval": 10,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved STI-DDPM checkpoint on the test split.")
    parser.add_argument("--checkpoint_path", "--checkpoint", required=True, help="Path to a checkpoint or run dir.")
    parser.add_argument("--data", default=None, help="Dataset name. Falls back to checkpoint metadata.")
    parser.add_argument("--model_name", default="sti_ddpm", help="Model name for result directory layout.")
    parser.add_argument("--result_dir", default="result", help="Root directory for evaluation artifacts.")
    parser.add_argument("--run_timestamp", default=None, help="Optional fixed timestamp for the output directory.")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device id. Use -1 for CPU.")
    parser.add_argument("--random_seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--fast_steps", type=int, default=None, help="Override reverse diffusion steps.")
    parser.add_argument("--guidance_scale", type=float, default=None, help="Override classifier-free guidance scale.")
    parser.add_argument("--mc_eval", type=int, default=None, help="Override Monte Carlo repeats.")
    args = parser.parse_args()
    if args.run_timestamp is None:
        args.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return args


def load_checkpoint(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def apply_checkpoint_config(args, checkpoint):
    checkpoint_args = dict(checkpoint.get("args") or {})
    if "w" in checkpoint_args and "guidance_scale" not in checkpoint_args:
        checkpoint_args["guidance_scale"] = checkpoint_args["w"]
    for key in CHECKPOINT_CONFIG_KEYS:
        if key in checkpoint_args:
            setattr(args, key, checkpoint_args[key])
    if args.data is None:
        args.data = checkpoint.get("dataset") or checkpoint_args.get("data")
    if args.data is None:
        raise ValueError("--data is required when the checkpoint does not contain a dataset name")
    for key, default_value in DEFAULT_EVAL_CONFIG.items():
        if getattr(args, key, None) is None:
            setattr(args, key, checkpoint_args.get(key, default_value))


def infer_fold_from_checkpoint_path(path: Path):
    for part in path.parts:
        if part.startswith("fold_"):
            suffix = part.split("fold_", 1)[1]
            if suffix.isdigit():
                return int(suffix)
    return None


def build_model(args, num_labels: int, numeric_feature_dim: int, device):
    return TableConditionalDiffusion(
        hf_model_name=args.shortcut_name,
        num_labels=num_labels,
        y_dim=args.y_dim,
        cond_dim=args.cond_dim,
        numeric_feature_dim=numeric_feature_dim,
        T=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        time_dim=args.time_dim,
        diffuser_type=args.diffuser_type,
        hidden=args.hidden_dim,
        cond_drop_prob=args.cond_drop_prob,
        beta_schedule=args.beta_schedule,
    ).to(device)


def infer_numeric_feature_dim(checkpoint, default: int = 6):
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    weight = state_dict.get("encoder.numeric_encoder.net.0.weight")
    if weight is None or weight.ndim != 2:
        return default
    return int(weight.shape[1])


def patch_numeric_encoder_for_checkpoint(model):
    base_encoder = model.encoder.numeric_encoder
    expected_dim = base_encoder.net[0].in_features

    class CompatibleNumericEncoder(nn.Module):
        def __init__(self, wrapped_encoder, expected_dim):
            super().__init__()
            self.wrapped_encoder = wrapped_encoder
            self.expected_dim = expected_dim

        def forward(self, numeric_features):
            current_dim = numeric_features.size(-1)
            if current_dim < self.expected_dim:
                numeric_features = F.pad(numeric_features, (0, self.expected_dim - current_dim))
            elif current_dim > self.expected_dim:
                numeric_features = numeric_features[..., : self.expected_dim]
            return self.wrapped_encoder(numeric_features)

    model.encoder.numeric_encoder = CompatibleNumericEncoder(base_encoder, expected_dim)


def resolve_checkpoint_targets(checkpoint_path: Path):
    if checkpoint_path.is_dir():
        fold_dirs = sorted(path for path in checkpoint_path.glob("fold_*") if path.is_dir())
        if fold_dirs:
            targets = []
            for fold_dir in fold_dirs:
                for candidate in (
                    fold_dir / "best" / "micro" / "model.pt",
                    fold_dir / "best" / "macro" / "model.pt",
                    fold_dir / "best" / "model.pt",
                ):
                    if candidate.exists():
                        targets.append(candidate)
                        break
                else:
                    raise FileNotFoundError(f"could not find a best checkpoint under {fold_dir}")
            return targets
        for candidate in (
            checkpoint_path / "best" / "micro" / "model.pt",
            checkpoint_path / "best" / "macro" / "model.pt",
            checkpoint_path / "best" / "model.pt",
        ):
            if candidate.exists():
                return [candidate]
        raise FileNotFoundError(f"could not resolve a checkpoint under {checkpoint_path}")
    return [checkpoint_path]


def evaluate_single_checkpoint(args, checkpoint_path: Path, device):
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    apply_checkpoint_config(args, checkpoint)
    numeric_feature_dim = infer_numeric_feature_dim(checkpoint)

    checkpoint_dataset = checkpoint.get("dataset")
    if checkpoint_dataset and checkpoint_dataset != args.data:
        print(f"warning: checkpoint dataset={checkpoint_dataset}, requested data={args.data}")

    split_mode = resolve_split_mode(args.data, args.split_mode)
    if split_mode == "cv" and args.cv_fold == -1:
        inferred_fold = infer_fold_from_checkpoint_path(checkpoint_path)
        if inferred_fold is None:
            raise ValueError("CV evaluation requires a fold in the checkpoint path or metadata")
        args.cv_fold = inferred_fold

    tokenizer = build_tokenizer(args.shortcut_name)
    split_mode, train_df, valid_df, test_df, num_labels = build_dataframes(args)
    checkpoint_num_labels = int(checkpoint.get("num_labels", num_labels))
    if checkpoint_num_labels != num_labels:
        print(
            f"warning: checkpoint num_labels={checkpoint_num_labels}, current split num_labels={num_labels}; using checkpoint value"
        )

    (
        _train_dataset,
        _valid_dataset,
        _test_dataset,
        _train_dataloader,
        _valid_dataloader,
        test_dataloader,
    ) = build_dataloaders(args, tokenizer, train_df, valid_df, test_df)

    model = build_model(args, checkpoint_num_labels, numeric_feature_dim, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    patch_numeric_encoder_for_checkpoint(model)
    model.eval()

    test_stats = evaluate_f1_stats(
        model,
        test_dataloader,
        device,
        fast_steps=args.fast_steps,
        guidance_scale=args.guidance_scale,
        seed=args.random_seed,
        mc=args.mc_eval,
    )
    return {
        "checkpoint": checkpoint,
        "checkpoint_path": str(checkpoint_path),
        "split_mode": split_mode,
        "fold": args.cv_fold if split_mode == "cv" else None,
        "test_stats": test_stats,
        "dataset": args.data,
    }


def resolve_device(cuda_index: int) -> torch.device:
    use_cuda = cuda_index >= 0 and torch.cuda.is_available()
    return torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")


def summarize_results(results):
    if len(results) == 1:
        metric_name = results[0]["checkpoint"].get("selection_metric", "checkpoint")
        return {metric_name: results[0]["test_stats"]}

    average_micro = sum(result["test_stats"]["micro_f1"] for result in results) / len(results)
    average_macro = sum(result["test_stats"]["macro_f1"] for result in results) / len(results)
    total_eval = sum(result["test_stats"]["n_eval"] for result in results)
    return {
        "per_checkpoint": [
            {
                "checkpoint_path": result["checkpoint_path"],
                "fold": result["fold"],
                "selection_metric": result["checkpoint"].get("selection_metric", "checkpoint"),
                "test": result["test_stats"],
            }
            for result in results
        ],
        "average": {
            "micro_f1": average_micro,
            "macro_f1": average_macro,
            "n_eval": total_eval,
        },
    }


def main():
    args = parse_args()
    setup_seed(args.random_seed)

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    device = resolve_device(args.cuda)
    targets = resolve_checkpoint_targets(checkpoint_path)
    results = []
    for target in targets:
        eval_args = argparse.Namespace(**vars(args))
        results.append(evaluate_single_checkpoint(eval_args, target, device))

    if not results:
        raise ValueError("no checkpoints were resolved for evaluation")

    args.data = results[0]["dataset"]
    split_mode = results[0]["split_mode"]
    fold = results[0]["fold"] if len(results) == 1 else None
    run_dir = make_run_dir(args, split_mode, fold=fold)
    test_results = summarize_results(results)

    save_json(run_dir / "hparams.json", vars(args))
    save_json(run_dir / "test_metrics.json", test_results)
    save_json(
        run_dir / "metrics.json",
        {
            "dataset": args.data,
            "model_name": args.model_name,
            "run_timestamp": args.run_timestamp,
            "split_mode": split_mode,
            "fold": fold,
            "checkpoint_path": str(checkpoint_path),
            "test": test_results,
        },
    )

    if len(results) == 1:
        metric_name = results[0]["checkpoint"].get("selection_metric", "checkpoint")
        metrics = results[0]["test_stats"]
        print(
            f"TEST[{metric_name}] micro-F1 {metrics['micro_f1']:.6f} macro-F1 {metrics['macro_f1']:.6f} n_eval {metrics['n_eval']} -> {run_dir}"
        )
        return

    average = test_results["average"]
    print(
        f"TEST[{len(results)} checkpoints] micro-F1 {average['micro_f1']:.6f} macro-F1 {average['macro_f1']:.6f} n_eval {average['n_eval']} -> {run_dir}"
    )


if __name__ == "__main__":
    main()
