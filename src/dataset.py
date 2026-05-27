import ast
import operator
import random
from functools import reduce
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

PAD_ID = 0


def _to_numeric_values(cells: List[str]) -> List[float]:
    values = []
    for cell in cells:
        text = str(cell).strip().replace(",", "")
        if not text:
            continue
        try:
            value = float(text)
        except ValueError:
            continue
        if np.isfinite(value):
            values.append(value)
    return values


def _numeric_summary_stats(cells: List[str]) -> Tuple[List[float], float]:
    values = _to_numeric_values(cells)
    if not values:
        return [0.0] * 6, 0.0
    array = np.asarray(values, dtype=np.float32)
    unique_values, counts = np.unique(array, return_counts=True)
    mode_value = float(unique_values[np.argmax(counts)])
    return [
        float(array.mean()),
        float(array.std()),
        float(array.min()),
        float(array.max()),
        float(np.median(array)),
        mode_value,
    ], 1.0


def collate_fn(batch, pad_token_id: int = PAD_ID):
    input_ids = pad_sequence([item["data"] for item in batch], batch_first=True, padding_value=pad_token_id)
    attention_mask = (input_ids != pad_token_id).long()
    cls_indexes = pad_sequence([item["cls_indexes"] for item in batch], batch_first=True, padding_value=-1)
    col_label_ids_pad = pad_sequence([item["label"] for item in batch], batch_first=True, padding_value=-1)
    numeric_features = pad_sequence(
        [item["numeric_features"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )
    numeric_feature_mask = pad_sequence(
        [item["numeric_feature_mask"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )

    batch_size, max_cols = cls_indexes.shape
    col_table_ids = []
    col_pos_in_table = []
    col_label_ids = []
    for batch_index in range(batch_size):
        for column_index in range(max_cols):
            if cls_indexes[batch_index, column_index].item() < 0:
                continue
            col_table_ids.append(batch_index)
            col_pos_in_table.append(column_index)
            col_label_ids.append(int(col_label_ids_pad[batch_index, column_index].item()))

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "cls_indexes": cls_indexes,
        "col_table_ids": torch.tensor(col_table_ids, dtype=torch.long),
        "col_pos_in_table": torch.tensor(col_pos_in_table, dtype=torch.long),
        "col_label_ids": torch.tensor(col_label_ids, dtype=torch.long),
        "numeric_features": numeric_features,
        "numeric_feature_mask": numeric_feature_mask,
    }


def _split_cells(serialized: str) -> List[str]:
    if pd.isna(serialized):
        return []
    text = str(serialized).strip()
    if not text:
        return []
    if "[ROW]" in text:
        parts = text.split("[ROW]")
    elif ";" in text:
        parts = text.split(";")
    else:
        parts = [text]
    return [part.strip() for part in parts if part.strip()]


def _canonicalize_common_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    out = dataframe.copy()
    out["table_id"] = out["table_id"].astype(str)
    out["column_index"] = pd.to_numeric(out["column_index"], errors="coerce").fillna(-1).astype(int)
    out["data"] = out["data"].astype(str)
    out = out[out["column_index"] >= 0].copy()
    return out[["table_id", "column_index", "data", "label_id"]]


def _parse_presplit_label(value) -> int:
    if pd.isna(value):
        return -1
    text = str(value).strip()
    if text.startswith("["):
        try:
            vector = ast.literal_eval(text)
        except Exception:
            return -1
        for index, item in enumerate(vector):
            if item > 0:
                return index
        return -1
    try:
        return int(float(text))
    except Exception:
        return -1


def canonicalize_doduo_cv_df(dataframe: pd.DataFrame) -> pd.DataFrame:
    out = dataframe.copy()
    out["column_index"] = out["col_idx"]
    out["label_id"] = pd.to_numeric(out["class_id"], errors="coerce").fillna(-1).astype(int)
    return _canonicalize_common_columns(out)


def canonicalize_presplit_df(dataframe: pd.DataFrame) -> pd.DataFrame:
    out = dataframe.copy()
    if "column_id" in out.columns and "column_index" not in out.columns:
        out = out.rename(columns={"column_id": "column_index"})
    out["label_id"] = out["label"].map(_parse_presplit_label).astype(int)
    return _canonicalize_common_columns(out)


def split_by_table(dataframe: pd.DataFrame, train_ratio: float, valid_ratio: float, seed: int):
    if train_ratio <= 0 or valid_ratio < 0 or train_ratio + valid_ratio > 1:
        raise ValueError("invalid split ratios")
    table_ids = dataframe["table_id"].drop_duplicates().tolist()
    rng = random.Random(seed)
    rng.shuffle(table_ids)

    total_tables = len(table_ids)
    num_train = int(total_tables * train_ratio)
    num_valid = int(total_tables * valid_ratio)
    train_tables = set(table_ids[:num_train])
    valid_tables = set(table_ids[num_train : num_train + num_valid])
    test_tables = set(table_ids[num_train + num_valid :])

    train_df = dataframe[dataframe["table_id"].isin(train_tables)].copy()
    valid_df = dataframe[dataframe["table_id"].isin(valid_tables)].copy()
    test_df = dataframe[dataframe["table_id"].isin(test_tables)].copy()
    return train_df, valid_df, test_df


def _find_cv_prefix(base_dir: Path) -> str:
    candidates = sorted(base_dir.glob("*_cv_0.csv"))
    if not candidates:
        raise FileNotFoundError(f"no *_cv_0.csv files found in {base_dir}")
    return candidates[0].name[: -len("_cv_0.csv")]


def build_cv_splits(data_dir: str, cv_fold: int, valid_ratio: float, seed: int):
    base_dir = Path(data_dir)
    prefix = _find_cv_prefix(base_dir)
    test_path = base_dir / f"{prefix}_cv_{cv_fold}.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"missing cv file: {test_path}")

    train_parts = []
    for fold in range(5):
        if fold == cv_fold:
            continue
        fold_path = base_dir / f"{prefix}_cv_{fold}.csv"
        if not fold_path.exists():
            raise FileNotFoundError(f"missing cv file: {fold_path}")
        train_parts.append(canonicalize_doduo_cv_df(pd.read_csv(fold_path)))

    train_all = pd.concat(train_parts, axis=0, ignore_index=True)
    test_df = canonicalize_doduo_cv_df(pd.read_csv(test_path))
    train_df, valid_df, _ = split_by_table(
        train_all,
        train_ratio=1.0 - valid_ratio,
        valid_ratio=valid_ratio,
        seed=seed,
    )
    return train_df, valid_df, test_df


def remap_noncontiguous_labels(dataframes: List[pd.DataFrame]) -> Tuple[List[pd.DataFrame], int, Dict[int, int]]:
    label_values = set()
    for dataframe in dataframes:
        label_values.update(dataframe.loc[dataframe["label_id"] >= 0, "label_id"].astype(int).tolist())
    if not label_values:
        return dataframes, 0, {}

    sorted_ids = sorted(label_values)
    mapping = {old_id: new_id for new_id, old_id in enumerate(sorted_ids)}
    remapped = []
    for dataframe in dataframes:
        copy = dataframe.copy()
        copy["label_id"] = copy["label_id"].map(lambda value: mapping.get(int(value), -1)).astype(int)
        remapped.append(copy)
    return remapped, len(sorted_ids), mapping


class WikiTableWiseDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        canonical_df: pd.DataFrame,
        max_length: int = 512,
        max_cols: int = 16,
        max_rows_per_col: int = 32,
        adaptive_col_budget: bool = True,
    ):
        if canonical_df is None:
            raise ValueError("canonical_df is required")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_cols = max_cols
        self.max_rows_per_col = max_rows_per_col
        self.adaptive_col_budget = adaptive_col_budget

        dataframe = canonical_df.copy()
        required = {"table_id", "column_index", "data", "label_id"}
        if not required.issubset(dataframe.columns):
            raise ValueError(f"canonical_df must contain columns: {required}")

        dataframe["table_id"] = dataframe["table_id"].astype(str)
        dataframe["column_index"] = dataframe["column_index"].astype(int)
        dataframe["data"] = dataframe["data"].astype(str)
        dataframe["label_id"] = dataframe["label_id"].astype(int)
        dataframe = dataframe.sort_values(["table_id", "column_index"]).reset_index(drop=True)
        self.df = dataframe

        self.table_ids = []
        self.table_ranges = []
        current_table_id = None
        start_index = 0
        for index, table_id in enumerate(dataframe["table_id"].tolist()):
            if current_table_id is None:
                current_table_id = table_id
                start_index = index
                continue
            if table_id != current_table_id:
                self.table_ids.append(current_table_id)
                self.table_ranges.append((start_index, index))
                current_table_id = table_id
                start_index = index
        if current_table_id is not None:
            self.table_ids.append(current_table_id)
            self.table_ranges.append((start_index, len(dataframe)))

    def __len__(self):
        return len(self.table_ids)

    def _serialize_column(self, raw_text: str) -> str:
        cells = _split_cells(raw_text)
        if self.max_rows_per_col is not None and len(cells) > self.max_rows_per_col:
            cells = cells[: self.max_rows_per_col]
        if not cells:
            return "[EMPTY]"
        return " ".join(cells)

    def _select_columns(self, group_df: pd.DataFrame) -> pd.DataFrame:
        if len(group_df) <= self.max_cols:
            return group_df
        labeled = group_df[group_df["label_id"] >= 0]
        unlabeled = group_df[group_df["label_id"] < 0]
        if len(labeled) >= self.max_cols:
            selected = labeled.iloc[: self.max_cols]
        else:
            need = self.max_cols - len(labeled)
            selected = pd.concat([labeled, unlabeled.iloc[:need]], axis=0)
        return selected.sort_values("column_index")

    def _encode_table(self, group_df: pd.DataFrame) -> Dict[str, Any]:
        group_df = self._select_columns(group_df)
        column_texts = [self._serialize_column(value) for value in group_df["data"].tolist()]
        numeric_stats = []
        numeric_mask = []
        for raw_text in group_df["data"].tolist():
            cells = _split_cells(raw_text)
            if self.max_rows_per_col is not None and len(cells) > self.max_rows_per_col:
                cells = cells[: self.max_rows_per_col]
            stats, mask = _numeric_summary_stats(cells)
            numeric_stats.append(stats)
            numeric_mask.append(mask)
        label_ids = group_df["label_id"].tolist()

        if self.adaptive_col_budget:
            per_col_max_length = max(16, min(256, self.max_length // max(1, len(column_texts))))
        else:
            per_col_max_length = min(256, self.max_length)

        token_ids_list = []
        cls_indexes = []
        offset = 0
        for text in column_texts:
            token_ids = self.tokenizer.encode(
                str(text),
                add_special_tokens=False,
                max_length=max(1, per_col_max_length - 1),
                truncation=True,
            )
            column_ids = [self.tokenizer.cls_token_id] + token_ids
            cls_indexes.append(offset)
            token_ids_list.append(column_ids)
            offset += len(column_ids)

        table_token_ids = torch.LongTensor(reduce(operator.add, token_ids_list))
        return {
            "table_id": group_df["table_id"].iloc[0],
            "num_col": len(column_texts),
            "data": table_token_ids,
            "cls_indexes": torch.LongTensor(cls_indexes),
            "label": torch.LongTensor(label_ids),
            "numeric_features": torch.tensor(numeric_stats, dtype=torch.float32),
            "numeric_feature_mask": torch.tensor(numeric_mask, dtype=torch.float32),
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        start_index, end_index = self.table_ranges[index]
        group_df = self.df.iloc[start_index:end_index]
        return self._encode_table(group_df)
