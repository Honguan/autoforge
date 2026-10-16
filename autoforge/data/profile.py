"""Dataset profiling that respects the privacy guard.

- summary access: schema and aggregate statistics only (no rows). Top values are
  shown only for low-cardinality columns, which are categories, not identifiers.
- full access: the same plus a few sample rows.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from autoforge.policy.privacy import Access

MAX_ROWS = 200_000
LOW_CARDINALITY = 20
SAMPLE_ROWS = 5


def _num(v: str) -> float | None:
    try:
        return float(v)
    except ValueError:
        return None


def _profile_csv(path: Path, access: Access) -> dict:
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        cols = {h: {"values": Counter(), "missing": 0, "numeric": [], "non_numeric": 0} for h in header}
        samples, n, truncated = [], 0, False
        for row in reader:
            if n >= MAX_ROWS:
                truncated = True
                break
            n += 1
            if len(samples) < SAMPLE_ROWS:
                samples.append(dict(zip(header, row)))
            for h, v in zip(header, row):
                c = cols[h]
                if v == "":
                    c["missing"] += 1
                    continue
                if len(c["values"]) <= LOW_CARDINALITY or v in c["values"]:
                    c["values"][v] += 1
                x = _num(v)
                if x is None:
                    c["non_numeric"] += 1
                else:
                    c["numeric"].append(x)

    columns = {}
    for h, c in cols.items():
        present = n - c["missing"]
        is_numeric = present > 0 and c["non_numeric"] == 0
        info: dict = {"dtype": "numeric" if is_numeric else "categorical", "missing": c["missing"]}
        distinct = len(c["values"])
        if distinct <= LOW_CARDINALITY:
            info["distinct"] = distinct
            info["top_values"] = dict(c["values"].most_common(5))
        else:
            info["distinct"] = f">{LOW_CARDINALITY}"
        if is_numeric:
            xs = c["numeric"]
            info |= {"min": min(xs), "max": max(xs), "mean": round(sum(xs) / len(xs), 6)}
        columns[h] = info

    out = {"format": "csv", "rows": n, "rows_truncated_at": MAX_ROWS if truncated else None, "columns": columns}
    if access is Access.full:
        out["sample_rows"] = samples
    return out


def _profile_dir(path: Path) -> dict:
    exts: Counter = Counter()
    subdirs = sorted(p.name for p in path.iterdir() if p.is_dir())
    top_files = sorted(p.name for p in path.iterdir() if p.is_file())
    files = 0
    for p in path.rglob("*"):
        if p.is_file():
            files += 1
            exts[p.suffix.lower() or "(none)"] += 1
    return {
        "format": "directory",
        "files": files,
        "extensions": dict(exts.most_common(10)),
        "top_level_dirs": subdirs[:50],  # e.g. class folders in image datasets
        "top_level_files": top_files[:30],  # e.g. annotations.csv, metadata.csv, classes.txt
    }


def _profile_jsonl(path: Path) -> dict:
    """Record count and field names only (no values): enough to tell chat / preference / pair data apart."""
    import json

    rows, keys, bad = 0, Counter(), 0
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            rows += 1
            if rows > MAX_ROWS:
                break
            try:
                rec = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if isinstance(rec, dict):
                keys.update(rec.keys())
    return {"format": "jsonl", "records": min(rows, MAX_ROWS), "fields": dict(keys.most_common(30)),
            "invalid_lines": bad}


def profile(path: str | Path, access: Access) -> dict:
    p = Path(path).expanduser().resolve()
    base = {"path": str(p), "access": access.value}
    if p.is_dir():
        return base | _profile_dir(p)
    if p.suffix.lower() == ".csv":
        return base | _profile_csv(p, access)
    if p.suffix.lower() == ".jsonl":
        return base | _profile_jsonl(p)
    return base | {"format": p.suffix.lstrip(".") or "unknown", "bytes": p.stat().st_size}
