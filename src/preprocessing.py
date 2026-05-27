import argparse
import csv
import html
import os
import re
import unicodedata
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse

import pandas as pd

MISSING = "[MISSING]"
ROW_SEP = " [ROW] "
PLACEHOLDERS = {"", "nan", "none", "null", "n/a", "na"}

URL_RE = re.compile(r"(https?://[^\s;]+|www\.[^\s;]+)", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")
SPACE_RE = re.compile(r"\s+")
HEX_RE = re.compile(r"^0x[0-9a-f]{12,}$", re.IGNORECASE)
BIN_RE = re.compile(r"^b['\"].*\\x", re.IGNORECASE)
DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DATE_TIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[ t]\d{1,2}:\d{2}(:\d{2})?(\.\d+)?$")
ISO_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}t\d{2}:\d{2}:\d{2}(\.\d+)?([+-]\d{2}:?\d{2}|z)$", re.IGNORECASE)
CUSTOM_DT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}h\d{2}\.\d{2}(\.\d+)?$", re.IGNORECASE)
TIME_ONLY_RE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")
INT_RE = re.compile(r"^[+-]?\d+$")
FLOAT_RE = re.compile(r"^[+-]?\d+\.\d+$")
ALNUM_ID_RE = re.compile(r"^(?=.*[a-zA-Z])(?=.*\d)[A-Za-z0-9._:-]+$")
PATH_EXT_RE = re.compile(r"\.([A-Za-z0-9]{2,5})(?:$|[?#])")
LONG_NUM_RE = re.compile(r"\d{6,}")


def _normalize_space(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)
    text = CTRL_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def _bucket_ratio(value: float) -> str:
    if value < 0.2:
        return "LOW"
    if value < 0.6:
        return "MID"
    return "HIGH"


def _bucket_avg_len(value: float) -> str:
    if value < 8:
        return "SHORT"
    if value < 24:
        return "MID"
    return "LONG"


def _normalize_url(match: re.Match) -> str:
    raw = match.group(0)
    if raw.lower().startswith("www."):
        raw = "http://" + raw
    parsed = urlparse(raw)
    host = (parsed.netloc or "").lower()
    host = host.split("@")[-1]
    if ":" in host:
        host = host.split(":", 1)[0]
    depth = len([p for p in parsed.path.split("/") if p])
    suffix = PATH_EXT_RE.search(parsed.path or "")
    parts = ["[URL]"]
    if host:
        parts.append(host)
    parts.append(f"[PATH_{min(depth, 3) if depth else 0}]")
    if suffix:
        parts.append(f"[EXT_{suffix.group(1).lower()}]")
    return " ".join(parts)


def _normalize_datetime_token(text: str) -> Optional[str]:
    lower = text.lower()
    if text in {"1900-01-01", "1900-01-01 00:00:00", "1970-01-01", "1970-01-01 00:00:00"}:
        return "[DATE_SENTINEL]"
    if DATE_ONLY_RE.fullmatch(text):
        year = int(text[:4])
        return f"[DATE] [YEAR_{(year // 10) * 10}s] [MONTH_{text[5:7]}]"
    if DATE_TIME_RE.fullmatch(text) or ISO_TS_RE.fullmatch(lower) or CUSTOM_DT_RE.fullmatch(lower):
        year = int(text[:4])
        month = text[5:7]
        parts = [f"[DATE] [YEAR_{(year // 10) * 10}s] [MONTH_{month}] [HAS_TIME]"]
        if "." in text:
            parts.append("[HAS_MS]")
        if "t" in lower and ("+" in text[10:] or "-" in text[10:] or lower.endswith("z")):
            parts.append("[HAS_TZ]")
        return " ".join(parts)
    if TIME_ONLY_RE.fullmatch(text):
        return f"[TIME] [HH_{int(text.split(':', 1)[0]):02d}]"
    return None


def _normalize_numeric_token(text: str) -> Optional[str]:
    compact = text.strip()
    if INT_RE.fullmatch(compact):
        digits = len(compact.lstrip("+-"))
        if compact in {"0", "1"}:
            return f"[BIN_{compact}]"
        if digits >= 6:
            return "[LONG_NUMERIC_ID]"
        return f"[INT_{digits}D]"
    if FLOAT_RE.fullmatch(compact):
        frac = compact.split(".", 1)[1]
        parts = ["[FLOAT_NEG]" if compact.startswith("-") else "[FLOAT_POS]"]
        try:
            mag = abs(float(compact))
        except ValueError:
            mag = 0.0
        parts.append("[LT1]" if mag < 1 else "[GE1]")
        parts.append(f"[PREC_{len(frac)}]")
        return " ".join(parts)
    return None


def _normalize_free_text(text: str) -> str:
    text = TAG_RE.sub(" ", text)
    text = re.sub(r"\b(?:href|xmlns|ArrayOf\w+|format|description)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:String|Literal)\b", " ", text)
    text = re.sub(r"\b166-telephony\s*:\s*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\\r\\n|\\n|\\t", " ", text)
    text = SPACE_RE.sub(" ", text).strip()
    if not text:
        return MISSING
    return text


def normalize_cell_text(
    value,
    max_len: int = 64,
    normalize_urls: bool = True,
    normalize_numbers: bool = True,
    normalize_datetime: bool = True,
) -> str:
    if pd.isna(value):
        return MISSING
    text = _normalize_space(str(value))
    if text.lower() in PLACEHOLDERS:
        return MISSING
    if HEX_RE.fullmatch(text):
        return "[HEX_BLOB]"
    if BIN_RE.match(text) or "\\x" in text:
        return "[BINARY_BLOB]"
    if text.startswith("<?xml") or (text.startswith("<") and text.endswith(">") and "</" in text):
        return "[XML]"
    if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]") and ":" in text):
        return "[JSON]"
    if normalize_urls:
        text = URL_RE.sub(_normalize_url, text)
    dt_text = _normalize_datetime_token(text) if normalize_datetime else None
    if dt_text is not None:
        return dt_text
    num_text = _normalize_numeric_token(text) if normalize_numbers else None
    if num_text is not None:
        return num_text
    if ALNUM_ID_RE.fullmatch(text) and len(text) >= 4:
        return "[ALNUM_ID]"
    if LONG_NUM_RE.search(text) and len(text) > 16:
        return "[LONG_NUMERIC_ID]"
    suffix = PATH_EXT_RE.search(text)
    if ("\\" in text or "/" in text) and suffix:
        return f"[PATH] [EXT_{suffix.group(1).lower()}]"
    text = _normalize_free_text(text)
    if len(text) > max_len:
        text = text[:max_len].rstrip()
    return text or MISSING


def split_serialized_cells(serialized: str) -> List[str]:
    if pd.isna(serialized):
        return []
    text = str(serialized)
    if "[ROW]" in text:
        parts = text.split("[ROW]")
    elif ";" in text:
        parts = text.split(";")
    else:
        parts = [text]
    out = []
    for part in parts:
        cell = part.strip()
        if cell:
            out.append(cell)
    return out


def build_profile_text(values: Sequence[str]) -> str:
    if not values:
        return "[COL_META] [NUM_RATIO_LOW] [DATE_RATIO_LOW] [URL_RATIO_LOW] [UNIQUE_LOW] [AVG_LEN_SHORT]"
    non_missing = [v for v in values if v != MISSING]
    base = non_missing or list(values)
    total = max(len(base), 1)
    num_ratio = sum(v.startswith("[INT_") or v.startswith("[FLOAT_") or v.startswith("[BIN_") or v == "[LONG_NUMERIC_ID]" for v in base) / total
    date_ratio = sum(v.startswith("[DATE") or v.startswith("[TIME]") for v in base) / total
    url_ratio = sum("[URL]" in v or v.startswith("[PATH]") for v in base) / total
    unique_ratio = len(set(base)) / total
    avg_len = sum(len(v) for v in base) / total
    return " ".join(
        [
            "[COL_META]",
            f"[NUM_RATIO_{_bucket_ratio(num_ratio)}]",
            f"[DATE_RATIO_{_bucket_ratio(date_ratio)}]",
            f"[URL_RATIO_{_bucket_ratio(url_ratio)}]",
            f"[UNIQUE_{_bucket_ratio(unique_ratio)}]",
            f"[AVG_LEN_{_bucket_avg_len(avg_len)}]",
        ]
    )


def serialize_cells(
    values: Sequence[str],
    max_rows: int = 8,
    collapse_duplicates: bool = True,
) -> Tuple[str, str]:
    ordered = list(values)
    if collapse_duplicates:
        counts = {}
        first_seen = []
        for value in ordered:
            counts[value] = counts.get(value, 0) + 1
            if counts[value] == 1:
                first_seen.append(value)
        ordered = []
        for value in first_seen:
            ordered.append(value)
            repeat_count = counts[value] - 1
            if repeat_count > 0:
                ordered.append(f"[REPEATS_{repeat_count}]")
    if max_rows is not None and len(ordered) > max_rows:
        head_keep = max_rows // 2
        tail_keep = max_rows - head_keep
        ordered = ordered[:head_keep] + ordered[-tail_keep:]
    profile_text = build_profile_text(values)
    if not ordered:
        return MISSING, profile_text
    return ROW_SEP.join(ordered), profile_text


def normalize_serialized_column(
    raw_serialized: str,
    max_rows: int = 8,
    max_cell_len: int = 64,
    normalize_urls: bool = True,
    normalize_numbers: bool = True,
    normalize_datetime: bool = True,
    collapse_duplicates: bool = True,
) -> Tuple[str, str]:
    raw_cells = split_serialized_cells(raw_serialized)
    normalized = [
        normalize_cell_text(
            cell,
            max_len=max_cell_len,
            normalize_urls=normalize_urls,
            normalize_numbers=normalize_numbers,
            normalize_datetime=normalize_datetime,
        )
        for cell in raw_cells
    ]
    if not normalized:
        normalized = [MISSING]
    return serialize_cells(normalized, max_rows=max_rows, collapse_duplicates=collapse_duplicates)


def _normalize_column_values(
    serialized: str,
    max_rows_per_col: int,
    max_cell_len: int,
    collapse_duplicates: bool,
) -> Tuple[str, str]:
    return normalize_serialized_column(
        serialized,
        max_rows=max_rows_per_col,
        max_cell_len=max_cell_len,
        collapse_duplicates=collapse_duplicates,
    )


def _write_processed_row(
    writer: csv.writer,
    label_set: Set[str],
    table_id: str,
    label: str,
    column_index: int,
    serialized_data: str,
    max_rows_per_col: int,
    max_cell_len: int,
    collapse_duplicates: bool,
):
    if label != "NaN":
        label_set.add(label)
    data_str, profile_text = _normalize_column_values(
        serialized_data,
        max_rows_per_col=max_rows_per_col,
        max_cell_len=max_cell_len,
        collapse_duplicates=collapse_duplicates,
    )
    writer.writerow([table_id, label, int(column_index), data_str, profile_text])


def process_table_csv_dir(
    split_dir: str,
    out_csv_path: str,
    label_set: Set[str],
    max_rows_per_col: int,
    max_cell_len: int,
    collapse_duplicates: bool,
):
    files = [os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith(".csv")]
    with open(out_csv_path, "w", newline="", encoding="utf-8") as fw:
        writer = csv.writer(fw)
        writer.writerow(["table_id", "label", "column_index", "data", "profile_text"])
        for i, path in enumerate(files):
            table_id = os.path.splitext(os.path.basename(path))[0]
            try:
                df = pd.read_csv(path, dtype=str)
            except Exception:
                continue
            if df.shape[1] == 0:
                continue
            for col_idx, col_name in enumerate(df.columns):
                label = str(col_name).strip()
                _write_processed_row(
                    writer=writer,
                    label_set=label_set,
                    table_id=table_id,
                    label=label,
                    column_index=col_idx,
                    serialized_data=";".join(df[col_name].fillna(MISSING).astype(str).tolist()),
                    max_rows_per_col=max_rows_per_col,
                    max_cell_len=max_cell_len,
                    collapse_duplicates=collapse_duplicates,
                )
            if (i + 1) % 2000 == 0:
                print(f"{split_dir}: {i + 1}/{len(files)}")
    print(f"saved: {out_csv_path}")


def process_gt_cv_dir(
    data_dir: str,
    out_csv_path: str,
    label_set: Set[str],
    max_rows_per_col: int,
    max_cell_len: int,
    collapse_duplicates: bool,
):
    frames = []
    for path in sorted(Path(data_dir).glob("*.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"no csv files found in {data_dir}")
    df = pd.concat(frames, axis=0, ignore_index=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as fw:
        writer = csv.writer(fw)
        writer.writerow(["table_id", "label", "column_index", "data", "profile_text"])
        for _, row in df.iterrows():
            table_id = str(row["table_id"])
            label = str(row["class"]).strip()
            _write_processed_row(
                writer=writer,
                label_set=label_set,
                table_id=table_id,
                label=label,
                column_index=int(row["col_idx"]),
                serialized_data=row["data"],
                max_rows_per_col=max_rows_per_col,
                max_cell_len=max_cell_len,
                collapse_duplicates=collapse_duplicates,
            )
    print(f"saved: {out_csv_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Dataset name")
    parser.add_argument("--max_rows_per_col", type=int, default=8)
    parser.add_argument("--max_cell_len", type=int, default=64)
    parser.add_argument("--collapse_duplicates", type=bool, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    data = args.data
    out_dir = Path("datas") / data / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_labels: Set[str] = set()

    if data.startswith("gt"):
        data_dir = Path("data") / data
        process_gt_cv_dir(
            str(data_dir),
            str(out_dir / "columns.csv"),
            all_labels,
            max_rows_per_col=args.max_rows_per_col,
            max_cell_len=args.max_cell_len,
            collapse_duplicates=args.collapse_duplicates,
        )
    else:
        split_dir = Path("datas") / data
        if not split_dir.exists():
            raise FileNotFoundError(f"missing dataset directory: {split_dir}")
        process_table_csv_dir(
            str(split_dir),
            str(out_dir / "columns.csv"),
            all_labels,
            max_rows_per_col=args.max_rows_per_col,
            max_cell_len=args.max_cell_len,
            collapse_duplicates=args.collapse_duplicates,
        )

    with open(out_dir / "labels.txt", "w", encoding="utf-8") as f:
        for idx, label in enumerate(sorted(all_labels)):
            f.write(f"{idx}\t{label}\n")
    print("labels.txt saved")


if __name__ == "__main__":
    main()
