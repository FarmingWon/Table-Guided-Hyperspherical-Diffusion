import random
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from args import parse_args
from dataset import (
    WikiTableWiseDataset,
    build_cv_splits,
    canonicalize_presplit_df,
    collate_fn,
    remap_noncontiguous_labels,
    split_by_table,
)
from model import TableConditionalDiffusion
from utils import evaluate_f1_stats, make_run_dir, save_json


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(cuda_index: int) -> torch.device:
    use_cuda = cuda_index >= 0 and torch.cuda.is_available()
    return torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")


def make_optimizer(args, model):
    encoder_params = list(model.encoder.backbone.parameters())
    encoder_param_ids = {id(param) for param in encoder_params}
    other_params = [param for param in model.parameters() if id(param) not in encoder_param_ids]
    parameter_groups = [
        {"params": encoder_params, "lr": args.encoder_lr},
        {"params": other_params, "lr": args.lr},
    ]
    if args.optimizer == "adam":
        return torch.optim.Adam(parameter_groups, eps=1e-8, weight_decay=args.l2_decay)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(parameter_groups, eps=1e-8, weight_decay=args.l2_decay)
    if args.optimizer == "adagrad":
        return torch.optim.Adagrad(parameter_groups, eps=1e-8, weight_decay=args.l2_decay)
    if args.optimizer == "rmsprop":
        return torch.optim.RMSprop(parameter_groups, eps=1e-8, weight_decay=args.l2_decay)
    raise ValueError(f"unsupported optimizer: {args.optimizer}")


def resolve_hf_model_name(shortcut_name: str) -> str:
    if shortcut_name == "bert":
        return "bert-base-uncased"
    return shortcut_name


def build_tokenizer(shortcut_name: str):
    tokenizer = AutoTokenizer.from_pretrained(resolve_hf_model_name(shortcut_name), use_fast=False)
    if not isinstance(getattr(tokenizer, "_special_tokens_map", None), dict):
        tokenizer._special_tokens_map = {
            "unk_token": getattr(tokenizer, "unk_token", "[UNK]"),
            "sep_token": getattr(tokenizer, "sep_token", "[SEP]"),
            "pad_token": getattr(tokenizer, "pad_token", "[PAD]"),
            "cls_token": getattr(tokenizer, "cls_token", "[CLS]"),
            "mask_token": getattr(tokenizer, "mask_token", "[MASK]"),
            "additional_special_tokens": [],
        }
    return tokenizer


def resolve_split_mode(data_name: str, split_mode: str) -> str:
    if split_mode in {"random", "cv", "presplit"}:
        return split_mode
    data_dir = Path("data") / data_name
    if any(data_dir.glob("*_cv_*.csv")):
        return "cv"
    if (data_dir / "train.csv").exists() and (data_dir / "test.csv").exists():
        return "presplit"
    return "random"


def _finalize_splits(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame):
    (train_df, valid_df, test_df), num_labels, remap = remap_noncontiguous_labels(
        [train_df, valid_df, test_df]
    )
    if num_labels <= 0:
        raise ValueError("no labeled columns were found in the prepared splits")
    return train_df, valid_df, test_df, num_labels, remap


def build_dataframes(args):
    split_mode = resolve_split_mode(args.data, args.split_mode)
    if split_mode == "cv":
        data_dir = Path("data") / args.data
        train_df, valid_df, test_df = build_cv_splits(
            data_dir=str(data_dir),
            cv_fold=args.cv_fold,
            valid_ratio=args.valid_ratio,
            seed=args.random_seed,
        )
        train_df, valid_df, test_df, num_labels, remap = _finalize_splits(train_df, valid_df, test_df)
        print(f"split_mode=cv fold={args.cv_fold} num_labels={num_labels} remapped_labels={len(remap) > 0}")
        return split_mode, train_df, valid_df, test_df, num_labels
    if split_mode == "presplit":
        data_dir = Path("data") / args.data
        split_frames = {}
        for split_name in ("train", "valid", "test"):
            split_path = data_dir / f"{split_name}.csv"
            if not split_path.exists():
                raise FileNotFoundError(f"missing {split_path} for presplit mode")
            split_frames[split_name] = canonicalize_presplit_df(pd.read_csv(split_path))
        train_df, valid_df, test_df = (
            split_frames["train"],
            split_frames["valid"],
            split_frames["test"],
        )
        train_df, valid_df, test_df, num_labels, remap = _finalize_splits(train_df, valid_df, test_df)
        print(
            f"split_mode=presplit train={len(train_df)} valid={len(valid_df)} "
            f"test={len(test_df)} num_labels={num_labels} remapped_labels={len(remap) > 0}"
        )
        return split_mode, train_df, valid_df, test_df, num_labels
    data_dir = Path("data") / args.data
    csv_candidates = [path for path in sorted(data_dir.glob("*.csv")) if "_cv_" not in path.name]
    if not csv_candidates:
        raise FileNotFoundError(f"no suitable CSV found in {data_dir}")
    full_df = canonicalize_presplit_df(pd.read_csv(csv_candidates[0]))
    train_ratio = 1.0 - (args.valid_ratio + args.test_ratio)
    if train_ratio <= 0:
        raise ValueError("valid_ratio + test_ratio must be < 1.0")
    train_df, valid_df, test_df = split_by_table(
        full_df,
        train_ratio=train_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.random_seed,
    )
    train_df, valid_df, test_df, num_labels, _ = _finalize_splits(train_df, valid_df, test_df)
    print(f"split_mode=random num_labels={num_labels}")
    return split_mode, train_df, valid_df, test_df, num_labels


def build_dataloaders(args, tokenizer, train_df, valid_df, test_df):
    def make_dataset(dataframe):
        return WikiTableWiseDataset(
            tokenizer=tokenizer,
            max_length=args.max_length,
            max_cols=args.max_cols,
            adaptive_col_budget=args.adaptive_col_budget,
            canonical_df=dataframe,
        )

    train_dataset = make_dataset(train_df)
    valid_dataset = make_dataset(valid_df)
    test_dataset = make_dataset(test_df)

    def make_loader(dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=lambda batch: collate_fn(batch, pad_token_id=tokenizer.pad_token_id),
        )

    return (
        train_dataset,
        valid_dataset,
        test_dataset,
        make_loader(train_dataset, shuffle=True),
        make_loader(valid_dataset, shuffle=False),
        make_loader(test_dataset, shuffle=False),
    )


def _make_best_record():
    return {"epoch": -1, "micro_f1": -1.0, "macro_f1": -1.0, "n_eval": 0}


def _save_best_checkpoint(
    best_root: Path,
    metric_name: str,
    epoch: int,
    args,
    split_mode: str,
    num_labels: int,
    model,
    optimizer,
    best_stats,
):
    best_path = best_root / metric_name
    best_path.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "dataset": args.data,
        "split_mode": split_mode,
        "num_labels": num_labels,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "best_val": best_stats,
        "selection_metric": metric_name,
    }
    torch.save(checkpoint, best_path / "model.pt")
    save_json(best_path / "best_val_metrics.json", best_stats)


def main():
    args = parse_args()
    args.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_seed(args.random_seed)
    device = resolve_device(args.cuda)
    tokenizer = build_tokenizer(args.shortcut_name)
    split_mode = resolve_split_mode(args.data, args.split_mode)
    fold_values = range(5) if split_mode == "cv" and args.cv_fold == -1 else [args.cv_fold]
    cv_results = {}

    for fold in fold_values:
        args.cv_fold = fold
        split_mode, train_df, valid_df, test_df, num_labels = build_dataframes(args)
        (
            train_dataset,
            valid_dataset,
            test_dataset,
            train_dataloader,
            valid_dataloader,
            test_dataloader,
        ) = build_dataloaders(args, tokenizer, train_df, valid_df, test_df)

        model = TableConditionalDiffusion(
            hf_model_name=args.shortcut_name,
            num_labels=num_labels,
            y_dim=args.y_dim,
            cond_dim=args.cond_dim,
            T=args.timesteps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            time_dim=args.time_dim,
            diffuser_type=args.diffuser_type,
            hidden=args.hidden_dim,
            cond_drop_prob=args.cond_drop_prob,
            beta_schedule=args.beta_schedule,
        ).to(device)
        optimizer = make_optimizer(args, model)
        run_fold = args.cv_fold if split_mode == "cv" else None
        run_dir = make_run_dir(args, split_mode, fold=run_fold)
        best_dir = run_dir / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        save_json(run_dir / "hparams.json", vars(args))

        print(
            f"device={device} train_tables={len(train_dataset)} "
            f"valid_tables={len(valid_dataset)} test_tables={len(test_dataset)}"
        )

        best = {"micro": _make_best_record(), "macro": _make_best_record()}

        for epoch in range(args.epoch):
            freeze_backbone = epoch < args.freeze_encoder_epochs
            for parameter in model.encoder.backbone.parameters():
                parameter.requires_grad = not freeze_backbone

            model.train()
            loss_total = 0.0

            for batch in tqdm(train_dataloader, desc=f"epoch {epoch}"):
                batch = {key: value.to(device) for key, value in batch.items()}
                loss = model.forward_loss(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    cls_indexes=batch["cls_indexes"],
                    col_table_ids=batch["col_table_ids"],
                    col_pos_in_table=batch["col_pos_in_table"],
                    col_label_ids=batch["col_label_ids"],
                    numeric_features=batch["numeric_features"],
                    numeric_feature_mask=batch["numeric_feature_mask"],
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_total += loss.item()

            mean_loss = loss_total / max(1, len(train_dataloader))
            should_validate = ((epoch + 1) % args.eval_every == 0) or ((epoch + 1) == args.epoch)
            if not should_validate:
                print(f"epoch {epoch + 1} loss {mean_loss:.4f}")
                continue

            val_stats = evaluate_f1_stats(
                model,
                valid_dataloader,
                device,
                fast_steps=args.fast_steps,
                guidance_scale=args.guidance_scale,
                seed=args.random_seed,
                mc=args.mc_eval,
            )
            current_best = {
                "epoch": epoch,
                "micro_f1": float(val_stats["micro_f1"]),
                "macro_f1": float(val_stats["macro_f1"]),
                "n_eval": int(val_stats["n_eval"]),
            }
            updated_metrics = []

            if current_best["micro_f1"] > best["micro"]["micro_f1"]:
                best["micro"] = current_best.copy()
                _save_best_checkpoint(
                    best_dir,
                    "micro",
                    epoch,
                    args,
                    split_mode,
                    num_labels,
                    model,
                    optimizer,
                    best["micro"],
                )
                updated_metrics.append(f"micro-F1 {val_stats['micro_f1']:.6f}")

            if current_best["macro_f1"] > best["macro"]["macro_f1"]:
                best["macro"] = current_best.copy()
                _save_best_checkpoint(
                    best_dir,
                    "macro",
                    epoch,
                    args,
                    split_mode,
                    num_labels,
                    model,
                    optimizer,
                    best["macro"],
                )
                updated_metrics.append(f"macro-F1 {val_stats['macro_f1']:.6f}")

            if updated_metrics:
                print(f"new best at epoch {epoch + 1} | " + " | ".join(updated_metrics))

            save_json(
                run_dir / "metrics.json",
                {
                    "dataset": args.data,
                    "model_name": args.model_name,
                    "run_timestamp": args.run_timestamp,
                    "split_mode": split_mode,
                    "fold": args.cv_fold if split_mode == "cv" else None,
                    "best_val": best,
                },
            )
            print(
                f"epoch {epoch + 1} diffusion micro-F1 {val_stats['micro_f1']:.4f} "
                f"macro-F1 {val_stats['macro_f1']:.4f} n_eval {val_stats['n_eval']} loss {mean_loss:.4f}"
            )

        test_results = {}
        for metric_name in ("micro", "macro"):
            checkpoint_path = best_dir / metric_name / "model.pt"
            if not checkpoint_path.exists():
                continue
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
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
            test_results[metric_name] = test_stats
            print(
                f"TEST[{metric_name}] diffusion micro-F1 {test_stats['micro_f1']:.6f} "
                f"macro-F1 {test_stats['macro_f1']:.6f} n_eval {test_stats['n_eval']}"
            )
            save_json(run_dir / f"test_metrics_{metric_name}.json", test_stats)

        save_json(run_dir / "test_metrics.json", test_results)
        save_json(
            run_dir / "metrics.json",
            {
                "dataset": args.data,
                "model_name": args.model_name,
                "run_timestamp": args.run_timestamp,
                "split_mode": split_mode,
                "fold": args.cv_fold if split_mode == "cv" else None,
                "best_val": best,
                "test": test_results,
            },
        )

        if split_mode == "cv":
            cv_results[f"fold_{fold}"] = {
                "micro_best_on_test": {
                    "micro_f1": float(test_results["micro"]["micro_f1"]) if "micro" in test_results else None,
                    "macro_f1": float(test_results["micro"]["macro_f1"]) if "micro" in test_results else None,
                },
                "macro_best_on_test": {
                    "micro_f1": float(test_results["macro"]["micro_f1"]) if "macro" in test_results else None,
                    "macro_f1": float(test_results["macro"]["macro_f1"]) if "macro" in test_results else None,
                },
            }

    if split_mode == "cv" and len(cv_results) > 1:
        base_dir = Path(args.result_dir) / args.model_name / args.data / args.run_timestamp
        micro_best_count = sum(
            1 for result in cv_results.values() if result["micro_best_on_test"]["micro_f1"] is not None
        )
        macro_best_count = sum(
            1 for result in cv_results.values() if result["macro_best_on_test"]["macro_f1"] is not None
        )
        save_json(
            base_dir / "cv_summary.json",
            {
                "dataset": args.data,
                "run_timestamp": args.run_timestamp,
                "per_fold": cv_results,
                "avg_micro_best_micro_f1": sum(
                    result["micro_best_on_test"]["micro_f1"]
                    for result in cv_results.values()
                    if result["micro_best_on_test"]["micro_f1"] is not None
                )
                / max(1, micro_best_count),
                "avg_micro_best_macro_f1": sum(
                    result["micro_best_on_test"]["macro_f1"]
                    for result in cv_results.values()
                    if result["micro_best_on_test"]["macro_f1"] is not None
                )
                / max(1, micro_best_count),
                "avg_macro_best_micro_f1": sum(
                    result["macro_best_on_test"]["micro_f1"]
                    for result in cv_results.values()
                    if result["macro_best_on_test"]["micro_f1"] is not None
                )
                / max(1, macro_best_count),
                "avg_macro_best_macro_f1": sum(
                    result["macro_best_on_test"]["macro_f1"]
                    for result in cv_results.values()
                    if result["macro_best_on_test"]["macro_f1"] is not None
                )
                / max(1, macro_best_count),
            },
        )


if __name__ == "__main__":
    main()
