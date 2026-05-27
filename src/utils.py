import json
from datetime import datetime
from pathlib import Path

from sklearn.metrics import f1_score


def _collect_eval_predictions(
    model,
    dataloader,
    device,
    fast_steps=50,
    guidance_scale=3.0,
    seed=None,
    mc=1,
):
    all_preds = []
    all_labels = []
    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        pred_ids, _ = model.predict_labels(
            batch["input_ids"],
            batch["attention_mask"],
            batch["cls_indexes"],
            batch["col_table_ids"],
            batch["col_pos_in_table"],
            numeric_features=batch["numeric_features"],
            numeric_feature_mask=batch["numeric_feature_mask"],
            topk=1,
            fast_steps=fast_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            mc=mc,
        )
        preds = pred_ids.squeeze(-1)
        labels = batch["col_label_ids"]
        mask = labels >= 0
        if mask.any():
            all_preds.extend(preds[mask].detach().cpu().tolist())
            all_labels.extend(labels[mask].detach().cpu().tolist())
    return all_preds, all_labels


def evaluate_f1_stats(model, dataloader, device, fast_steps=50, guidance_scale=3.0, seed=None, mc=1):
    model.eval()
    all_preds, all_labels = _collect_eval_predictions(
        model=model,
        dataloader=dataloader,
        device=device,
        fast_steps=fast_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        mc=mc,
    )
    num_labels = int(model.label_emb.num_embeddings)
    label_ids = list(range(num_labels))
    if not all_labels:
        return {
            "micro_f1": 0.0,
            "macro_f1": 0.0,
            "class_f1": [0.0 for _ in label_ids],
            "n_eval": 0,
        }
    return {
        "micro_f1": float(f1_score(all_labels, all_preds, average="micro")),
        "macro_f1": float(f1_score(all_labels, all_preds, average="macro")),
        "class_f1": f1_score(
            all_labels,
            all_preds,
            average=None,
            labels=label_ids,
            zero_division=0,
        ).tolist(),
        "n_eval": len(all_labels),
    }


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(obj, file, indent=2, ensure_ascii=False)


def make_run_dir(args, split_mode: str, fold=None) -> Path:
    timestamp = getattr(args, "run_timestamp", None) or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.result_dir) / args.model_name / args.data / timestamp
    if split_mode == "cv" and fold is not None:
        run_dir = run_dir / f"fold_{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
