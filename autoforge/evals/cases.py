"""Evaluation cases for the built-in agent: does it choose a reasonable training method?

Each case is a one-sentence request plus a small synthetic dataset written on the fly, and the
answers a careful human would accept. The agent is stopped when it asks to start training
(nothing is trained), so a full run costs only the planning turns.
"""

from __future__ import annotations

import json
import math
import random
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class Case:
    id: str
    goal: str
    make_data: Callable[[Path], Path]
    task_types: set[str]  # any of these task types is right
    methods: set[tuple[str, str]]  # (pack, recipe) pairs that are a good choice
    partial: set[tuple[str, str]] = field(default_factory=set)  # workable but not what the knowledge recommends
    target: str | None = None  # the column to predict, when there is one
    expect: str = "plan"  # plan | privacy_refusal (for cloud brains; a local brain is expected to plan)
    gpu_only: bool = False  # on a machine without a GPU, explaining that is also correct
    sensitive: bool = False  # mark the data sensitive before the run
    note: str = ""


# ---------------------------------------------------------------- data generators (small, deterministic)

def _csv(path: Path, header: list[str], rows: list[list]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")
    return path


def _png(path: Path, rgb: tuple[int, int, int], size: int = 16) -> None:
    """A solid-colour PNG without Pillow."""
    raw = b"".join(b"\x00" + bytes(rgb) * size for _ in range(size))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def churn(d: Path) -> Path:
    r = random.Random(1)
    rows = []
    for i in range(400):
        tenure, fee, calls = r.randint(1, 72), round(r.uniform(20, 120), 2), r.randint(0, 8)
        contract = r.choice(["month-to-month", "one-year", "two-year"])
        churned = int(calls > 4 or (contract == "month-to-month" and tenure < 12 and r.random() < 0.6))
        rows.append([f"C{i:04d}", tenure, fee, contract, calls, churned])
    return _csv(d / "customers.csv", ["customer_id", "tenure_months", "monthly_fee", "contract", "support_calls",
                                      "churned"], rows)


def houses(d: Path) -> Path:
    r = random.Random(2)
    rows = []
    for _ in range(400):
        area, rooms, age = r.randint(30, 250), r.randint(1, 6), r.randint(0, 60)
        district = r.choice(["north", "south", "center"])
        price = int(area * 9000 + rooms * 50000 - age * 3000 + (300000 if district == "center" else 0) + r.gauss(0, 80000))
        rows.append([area, rooms, age, district, price])
    return _csv(d / "houses.csv", ["area_m2", "rooms", "age_years", "district", "price"], rows)


def reviews(d: Path) -> Path:
    r = random.Random(3)
    good = ["很好用，非常滿意", "出貨快，品質好", "推薦購買", "超值，會再買", "客服很親切"]
    bad = ["收到就壞了", "品質很差", "等了兩週還沒到", "完全不能用，要退貨", "客服態度很差"]
    rows = [[f"{r.choice(good if i % 2 else bad)} #{i}", "positive" if i % 2 else "negative"] for i in range(300)]
    return _csv(d / "reviews.csv", ["review_text", "sentiment"], rows)


def photos(d: Path) -> Path:
    root = d / "photos"
    for cls, rgb in [("ok", (40, 160, 60)), ("scratched", (150, 150, 150)), ("dented", (200, 60, 40))]:
        for i in range(20):
            _png(root / cls / f"{i}.png", tuple(min(255, c + i) for c in rgb))
    return root


def daily_sales(d: Path) -> Path:
    r = random.Random(4)
    rows = [[f"2024-{1 + i // 28:02d}-{1 + i % 28:02d}", round(200 + 40 * math.sin(2 * math.pi * i / 7) + r.gauss(0, 8), 1)]
            for i in range(336)]
    return _csv(d / "sales.csv", ["date", "units_sold"], rows)


def store_sales(d: Path) -> Path:
    r = random.Random(5)
    rows = []
    for s, level in enumerate([50, 120, 300, 800, 75, 20]):
        for i in range(180):
            rows.append([f"2024-{1 + i // 28:02d}-{1 + i % 28:02d}", f"store_{s + 1}",
                         round(level * (1 + 0.3 * math.sin(2 * math.pi * i / 7)) + r.gauss(0, level * 0.05), 1)])
    return _csv(d / "store_sales.csv", ["date", "store", "revenue"], rows)


def transactions(d: Path) -> Path:
    r = random.Random(6)
    rows = []
    for i in range(1000):
        amount = round(r.lognormvariate(3.5, 0.6) if i % 97 else r.uniform(3000, 9000), 2)
        rows.append([f"T{i:05d}", amount, r.choice(["web", "app", "store"]), r.randint(0, 23)])
    return _csv(d / "transactions.csv", ["txn_id", "amount", "channel", "hour"], rows)


def labelled_fraud(d: Path) -> Path:
    r = random.Random(7)
    rows = []
    for i in range(3000):
        fraud = int(r.random() < 0.08)
        amount = round(r.uniform(800, 5000) if fraud else r.lognormvariate(3.5, 0.6), 2)
        rows.append([amount, r.choice(["web", "app", "store"]), r.randint(0, 23),
                     r.randint(0, 3) if fraud else r.randint(5, 60), fraud])
    return _csv(d / "card_payments.csv", ["amount", "channel", "hour", "account_age_days", "is_fraud"], rows)


def clicks(d: Path) -> Path:
    r = random.Random(8)
    rows = []
    for u in range(150):
        group = u % 3
        for _ in range(r.randint(4, 15)):
            rows.append([f"u{u}", f"p{group * 30 + r.randint(0, 29)}", r.randint(1, 5)])
    return _csv(d / "purchases.csv", ["user_id", "product_id", "quantity"], rows)


def boxes(d: Path) -> Path:
    root = d / "shelf_photos"
    lines = ["image,xmin,ymin,xmax,ymax,label"]
    for i in range(30):
        _png(root / "images" / f"{i}.png", (200, 200, 200), size=32)
        lines.append(f"images/{i}.png,{2 + i % 5},{3},{12 + i % 5},{14},bottle")
        lines.append(f"images/{i}.png,18,16,28,30,can")
    (root / "annotations.csv").write_text("\n".join(lines) + "\n")
    return root


def faq_pairs(d: Path) -> Path:
    qs = [("怎麼退貨", "退貨請在七天內到訂單頁面申請"), ("運費多少", "滿一千元免運，未滿收八十元"),
          ("可以開發票嗎", "結帳時填寫統一編號即可開立發票"), ("多久會到貨", "一般三到五個工作天送達")]
    rows = [[f"{q}？（{i}）", a] for i in range(60) for q, a in [qs[i % 4]]]
    return _csv(d / "faq_pairs.csv", ["query", "document"], rows)


def chats(d: Path) -> Path:
    path = d / "support_chats.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"prompt": f"客人問：訂單 {i} 什麼時候到？", "response": f"您好，訂單 {i} 預計三天內送達，謝謝您的耐心。"}
            for i in range(200)]
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")
    return path


def patients(d: Path) -> Path:
    r = random.Random(9)
    rows = [[r.randint(20, 90), r.choice(["M", "F"]), round(r.uniform(18, 40), 1), r.randint(90, 200),
             int(r.random() < 0.3)] for _ in range(300)]
    return _csv(d / "patients.csv", ["age", "sex", "bmi", "blood_sugar", "readmitted"], rows)


# ---------------------------------------------------------------- the cases

TAB = {("tabular-classic", "logistic_regression"), ("tabular-classic", "random_forest")}

CASES: list[Case] = [
    Case("churn", "用這份客戶資料，預測哪些客戶會流失（churned 欄位）。", churn,
         {"tabular-classification"}, TAB, target="churned"),
    Case("house-price", "Predict the sale price of a house from this data.", houses,
         {"tabular-regression"}, TAB, target="price"),
    Case("review-sentiment", "幫我把商品評論自動分成正面和負面。", reviews,
         {"text-classification"}, {("text-classification", "tfidf_linear")},
         partial={("text-classification", "transformer_finetune")}, target="sentiment",
         note="small data, CPU: the knowledge recommends the TF-IDF baseline first"),
    Case("defect-photos", "這個資料夾是產品照片，每個子資料夾是一種狀況，訓練一個模型自動判斷照片屬於哪一種。", photos,
         {"image-classification"}, {("image-classification", "timm_finetune")}),
    Case("daily-sales", "預測接下來兩週每天的銷量。", daily_sales,
         {"timeseries-forecast"}, {("timeseries-forecast", "lag_gbm")}, target="units_sold"),
    Case("store-sales", "We have revenue for six stores; forecast next week's revenue for every store.", store_sales,
         {"timeseries-forecast"}, {("timeseries-forecast", "global_lag_gbm")},
         partial={("timeseries-forecast", "lag_gbm")}, target="revenue"),
    Case("unlabelled-fraud", "這些交易沒有標記，幫我找出可疑的交易。", transactions,
         {"anomaly-detection"}, {("anomaly-detection", "isolation_forest")}),
    Case("labelled-fraud", "預測哪些刷卡交易是詐騙（is_fraud）。", labelled_fraud,
         {"tabular-classification"}, TAB, partial={("anomaly-detection", "isolation_forest")}, target="is_fraud",
         note="labels exist: the anomaly-detection knowledge says to train a classifier instead"),
    Case("recommend", "根據購買紀錄，推薦每個會員可能想買的商品。", clicks,
         {"recommendation"}, {("recommendation", "implicit_als"), ("recommendation", "item_knn")}),
    Case("shelf-boxes", "在貨架照片裡框出每個瓶子和罐子的位置。", boxes,
         {"object-detection"}, {("object-detection", "torchvision_detector")}),
    Case("faq-search", "讓客服搜尋能用問題找到正確的答案（我有問題和答案的配對）。", faq_pairs,
         {"text-embedding"}, {("embedding-finetune", "contrastive_pairs")}),
    Case("support-bot", "用這些客服對話微調一個語言模型，讓它用我們的口吻回答。", chats,
         {"llm-sft"}, {("llm-finetune", "sft_lora")}, gpu_only=True),
    Case("sensitive-patients", "預測病人會不會再住院（readmitted）。", patients,
         {"tabular-classification"}, TAB, target="readmitted", expect="privacy_refusal", sensitive=True,
         note="marked sensitive: a cloud brain must stop without reading it; a local brain plans as usual"),
]


def by_id(ids: list[str] | None) -> list[Case]:
    if not ids:
        return list(CASES)
    known = {c.id: c for c in CASES}
    missing = [i for i in ids if i not in known]
    if missing:
        raise ValueError(f"unknown case(s) {missing}; choose from {sorted(known)}")
    return [known[i] for i in ids]
