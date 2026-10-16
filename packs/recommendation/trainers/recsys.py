"""Recommend items to users from an interaction log (implicit-feedback ALS or item-kNN), numpy/scipy only.

Data: one CSV, one row per interaction (purchase, click, view, rating):
  user_id,item_id[,weight][,timestamp]
Columns come from hyperparameters user_col / item_col / weight_col / time_col (ctx.target, if set, is used as
the weight column). If the default user/item columns are missing, common names are detected
(user, userId, customer_id / item, itemId, product_id, movieId; rating, count; timestamp, date).
Evaluation: leave-last-out per user (time_col) or one random held-out interaction per user (seed);
users with < 2 interactions stay in training only. After evaluation the model is refit on ALL data.
AutoForge controls: check_stop per ALS iteration (and per kNN stage), live `regularization` via run adjust,
train_loss + recall_at_k / ndcg_at_k per iteration, checkpoint per iteration (checkpoints/state.json,
resumed by `run retry`), best iteration kept.
"""

from __future__ import annotations

import json
from pathlib import Path

USER_NAMES = ["user_id", "user", "userid", "user_name", "customer_id", "customer", "customerid", "client_id",
              "visitor_id", "member_id"]
ITEM_NAMES = ["item_id", "item", "itemid", "product_id", "product", "productid", "movieid", "movie_id",
              "sku", "article_id", "song_id", "track_id", "book_id", "isbn"]
WEIGHT_NAMES = ["rating", "weight", "count", "clicks", "quantity", "qty", "plays", "score", "value"]
TIME_NAMES = ["timestamp", "time", "datetime", "date", "event_time", "ts", "created_at"]


# ---------------------------------------------------------------- data

def _pick(cols, wanted, default, candidates, role, required, log):
    """Use `wanted` if present; if it is the default (or null), detect a common name; else explain."""
    if isinstance(wanted, str) and wanted.lower() == "none" and not required:
        return None
    if wanted is not None and wanted in cols:
        return wanted
    if wanted is not None and wanted != default:
        raise ValueError(f"{role} column '{wanted}' not found; CSV columns are {list(cols)}. "
                         f"Set the hyperparameter {role}_col to one of them")
    lower = {str(c).lower(): c for c in cols}
    for cand in candidates:
        if cand in lower:
            log(f"using column '{lower[cand]}' as the {role} column")
            return lower[cand]
    if required:
        raise ValueError(f"no {role} column found (looked for '{default}' and {candidates[1:6]}...); "
                         f"CSV columns are {list(cols)}. Set the hyperparameter {role}_col")
    return None


def _load(ctx):
    import numpy as np
    import pandas as pd

    path = Path(ctx.data_path)
    if path.is_dir() or path.suffix.lower() not in (".csv", ".tsv", ".txt"):
        raise ValueError(f"data_path must be a CSV of interactions (user_id,item_id[,weight][,timestamp]), got {path}")
    if not path.is_file():
        raise FileNotFoundError(f"interaction CSV not found: {path}")
    df = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
    hp = ctx.hyperparameters
    ucol = _pick(df.columns, hp.get("user_col", "user_id"), "user_id", USER_NAMES, "user", True, ctx.log)
    icol = _pick(df.columns, hp.get("item_col", "item_id"), "item_id", ITEM_NAMES, "item", True, ctx.log)
    wcol = _pick(df.columns, hp.get("weight_col") or ctx.target, None, WEIGHT_NAMES, "weight", False, ctx.log)
    tcol = _pick(df.columns, hp.get("time_col"), None, TIME_NAMES, "time", False, ctx.log)
    chosen = [c for c in (ucol, icol, wcol, tcol) if c is not None]
    if len(set(chosen)) < len(chosen):
        raise ValueError(f"user/item/weight/time columns must differ, got {ucol}, {icol}, {wcol}, {tcol}")
    df = df.dropna(subset=[ucol, icol])
    if df.empty:
        raise ValueError(f"{path.name}: no rows with both '{ucol}' and '{icol}'")
    w = pd.to_numeric(df[wcol], errors="coerce") if wcol else pd.Series(1.0, index=df.index)
    if wcol and w.isna().all():
        raise ValueError(f"weight column '{wcol}' must be numeric (rating, count, clicks)")
    t = None
    if tcol:
        t = pd.to_numeric(df[tcol], errors="coerce")
        if t.isna().mean() > 0.5:
            t = pd.to_datetime(df[tcol], errors="coerce").astype("int64")
    frame = pd.DataFrame({"u": df[ucol].values, "i": df[icol].values, "w": w.fillna(0).values.astype(float),
                          "t": (t.fillna(0).values if t is not None else np.zeros(len(df)))})
    dropped = int((frame["w"] <= 0).sum())
    frame = frame[frame["w"] > 0]
    if dropped:
        ctx.log(f"dropped {dropped} rows with weight <= 0")
    frame = frame.groupby(["u", "i"], as_index=False).agg(w=("w", "sum"), t=("t", "max"))
    return frame, tcol is not None, (ucol, icol, wcol, tcol)


def _native(v):
    return v.item() if hasattr(v, "item") else v


def _split(frame, by_time, seed):
    """Return (train_frame, {user_idx: held_out_item_idx})."""
    import numpy as np

    rng = np.random.default_rng(seed)
    counts = frame.groupby("uidx")["iidx"].transform("size")
    cand = frame[counts >= 2]
    if by_time:
        held = cand.sort_values(["uidx", "t", "iidx"]).groupby("uidx").tail(1)
    else:
        held = cand.assign(r=rng.random(len(cand))).sort_values(["uidx", "r"]).groupby("uidx").head(1)
    test = dict(zip(held["uidx"].astype(int), held["iidx"].astype(int)))
    return frame.drop(index=held.index), test


def _csr(frame, n_users, n_items):
    import scipy.sparse as sp

    return sp.csr_matrix((frame["w"].values, (frame["uidx"].values, frame["iidx"].values)), shape=(n_users, n_items))


# ---------------------------------------------------------------- models

def _als_half(R, Y, reg, alpha):
    """Solve every row of X given Y with the Hu/Koren/Volinsky YtY trick (confidence = 1 + alpha*w)."""
    import numpy as np

    f = Y.shape[1]
    YtY = Y.T @ Y
    X = np.zeros((R.shape[0], f))
    eye = reg * np.eye(f)
    for u in range(R.shape[0]):
        lo, hi = R.indptr[u], R.indptr[u + 1]
        if lo == hi:
            continue
        idx, conf = R.indices[lo:hi], 1.0 + alpha * R.data[lo:hi]
        Yu = Y[idx]
        A = YtY + (Yu.T * (conf - 1.0)) @ Yu + eye
        X[u] = np.linalg.solve(A, Yu.T @ conf)
    return X


def _als_loss(R, X, Y, reg, alpha):
    import numpy as np

    coo = R.tocoo()
    s = np.einsum("ij,ij->i", X[coo.row], Y[coo.col])
    conf = 1.0 + alpha * coo.data
    total = np.sum((X.T @ X) * (Y.T @ Y)) + np.sum(conf * (1 - s) ** 2 - s ** 2)
    return float((total + reg * (np.sum(X * X) + np.sum(Y * Y))) / max(R.nnz, 1))


def _knn_similarity(R, neighbors, shrinkage):
    """Cosine item-item similarity with shrinkage, top-`neighbors` per item (row), as CSR."""
    import numpy as np
    import scipy.sparse as sp

    R = R.tocsc().astype(float)
    norms = np.sqrt(np.asarray(R.multiply(R).sum(axis=0))).ravel()
    norms[norms == 0] = 1.0
    Rn = R @ sp.diags(1.0 / norms)
    S = (Rn.T @ Rn).tocsr()
    if shrinkage > 0:  # damp similarities supported by few co-occurrences: n / (n + shrinkage)
        B = (R > 0).astype(float)
        F = (B.T @ B).tocsr()
        F.data = F.data / (F.data + shrinkage)
        S = S.multiply(F).tocsr()
    S = (S - sp.diags(S.diagonal())).tocsr()
    S.eliminate_zeros()
    rows, cols, vals = [], [], []
    for i in range(S.shape[0]):
        lo, hi = S.indptr[i], S.indptr[i + 1]
        if lo == hi:
            continue
        d, c = S.data[lo:hi], S.indices[lo:hi]
        if len(d) > neighbors:
            keep = np.argpartition(-d, neighbors - 1)[:neighbors]
            d, c = d[keep], c[keep]
        rows += [i] * len(d)
        cols += c.tolist()
        vals += d.tolist()
    return sp.csr_matrix((vals, (rows, cols)), shape=S.shape)


def _scores(model, users):
    """Dense score matrix for the given user indices."""
    import numpy as np

    if model["recipe"] == "implicit_als":
        return model["user_factors"][users] @ model["item_factors"].T
    return np.asarray((model["R"][users] @ model["similarity"]).todense())


def _top(scores, k, exclude=None):
    import numpy as np

    s = scores.astype(float).copy()
    if exclude is not None and len(exclude):
        s[exclude] = -np.inf
    k = min(k, int(np.isfinite(s).sum()))
    if k <= 0:
        return np.array([], dtype=int)
    idx = np.argpartition(-s, k - 1)[:k]
    return idx[np.argsort(-s[idx], kind="stable")]


def _evaluate(model, R, test, k, popularity=None):
    """recall@k (= hit rate with one held-out item), ndcg@k, coverage over test users."""
    import numpy as np

    users = np.array(sorted(test))
    if len(users) == 0:
        return {"recall_at_k": 0.0, "ndcg_at_k": 0.0, "coverage": 0.0}
    hits, ndcg, recommended = 0, 0.0, set()
    for start in range(0, len(users), 1024):
        chunk = users[start:start + 1024]
        S = np.tile(popularity, (len(chunk), 1)) if popularity is not None else _scores(model, chunk)
        for row, u in zip(S, chunk):
            seen = R.indices[R.indptr[u]:R.indptr[u + 1]]
            top = _top(row, k, seen)
            recommended.update(top.tolist())
            pos = np.where(top == test[u])[0]
            if len(pos):
                hits += 1
                ndcg += 1.0 / np.log2(pos[0] + 2)
    return {"recall_at_k": hits / len(users), "ndcg_at_k": float(ndcg / len(users)),
            "coverage": len(recommended) / R.shape[1]}


def _fit_als(ctx, R, iterations, test=None, k=10, start=None):
    """Run ALS. With `test`, evaluates + checkpoints each iteration and returns the best iteration."""
    import numpy as np

    hp = ctx.hyperparameters
    f, alpha = int(hp.get("factors", 64)), float(hp.get("alpha", 10.0))
    rng = np.random.default_rng(ctx.seed)
    X = rng.normal(0, 0.01, (R.shape[0], f))
    Y = rng.normal(0, 0.01, (R.shape[1], f))
    first, best = 0, None
    if start is not None:
        state = start
        ck = np.load(state["path"])
        X, Y, first = ck["X"], ck["Y"], state["iteration"] + 1
        best = state.get("best")
        ctx.log(f"resuming after iteration {state['iteration']}")
    Rt = R.T.tocsr()
    for it in range(first, iterations):
        ctx.check_stop()
        changed = ctx.check_updates()
        reg = float(hp.get("regularization", 0.1))
        if "regularization" in changed:
            ctx.log(f"regularization -> {reg}")
        X = _als_half(R, Y, reg, alpha)
        ctx.check_stop()
        Y = _als_half(Rt, X, reg, alpha)
        loss = _als_loss(R, X, Y, reg, alpha)
        if test is None:
            ctx.log(f"refit iteration {it + 1}/{iterations}: train_loss={loss:.4f}")
            continue
        ctx.log_metric("train_loss", loss, step=it)
        m = _evaluate({"recipe": "implicit_als", "user_factors": X, "item_factors": Y}, R, test, k)
        ctx.log_metric("recall_at_k", m["recall_at_k"], step=it)
        ctx.log_metric("ndcg_at_k", m["ndcg_at_k"], step=it)
        ck = ctx.checkpoint_dir / f"iter{it}.npz"
        np.savez(ck, X=X, Y=Y)
        if best is None or (m["recall_at_k"], m["ndcg_at_k"]) > (best["recall_at_k"], best["ndcg_at_k"]):
            best = dict(m, iteration=it)
        for old in ctx.checkpoint_dir.glob("iter*.npz"):
            if old != ck:
                old.unlink(missing_ok=True)
        (ctx.checkpoint_dir / "state.json").write_text(json.dumps({"iteration": it, "path": str(ck), "best": best}))
        ctx.log(f"iteration {it + 1}/{iterations}: train_loss={loss:.4f} recall@{k}={m['recall_at_k']:.4f} "
                f"ndcg@{k}={m['ndcg_at_k']:.4f}")
    return X, Y, best


# ---------------------------------------------------------------- training

def _train(ctx, recipe: str) -> dict:
    import joblib
    import numpy as np

    hp = ctx.hyperparameters
    k = int(hp.get("k", 10))
    frame, by_time, cols = _load(ctx)
    users, uidx = np.unique(frame["u"].values, return_inverse=True)
    items, iidx = np.unique(frame["i"].values, return_inverse=True)
    frame = frame.assign(uidx=uidx, iidx=iidx)
    nu, ni = len(users), len(items)
    if ni < 2 or nu < 2:
        raise ValueError(f"need at least 2 users and 2 items, got {nu} users and {ni} items")
    train_frame, test = _split(frame, by_time, ctx.seed)
    R = _csr(train_frame, nu, ni)
    ctx.log(f"{recipe}: {len(frame)} interactions, {nu} users, {ni} items; evaluating on {len(test)} users "
            f"({'last interaction by time' if by_time else 'one random interaction'} held out), k={k}")
    if not test:
        raise ValueError("no user has 2+ interactions, so nothing can be evaluated; recommendation needs repeat users")

    pop = np.asarray((R > 0).sum(axis=0)).ravel().astype(float)
    base = _evaluate(None, R, test, k, popularity=pop)
    ctx.log_metric("popularity_recall_at_k", base["recall_at_k"])
    ctx.log(f"popularity baseline: recall@{k}={base['recall_at_k']:.4f} ndcg@{k}={base['ndcg_at_k']:.4f}")

    if recipe == "implicit_als":
        iterations = int(hp.get("iterations", 15))
        state_file, start = ctx.latest_checkpoint(), None
        if state_file is not None and state_file.name == "state.json":
            start = json.loads(state_file.read_text())
        X, Y, best = _fit_als(ctx, R, iterations, test, k, start)
        metrics = {m: best[m] for m in ("recall_at_k", "ndcg_at_k", "coverage")}
        best_iters = best["iteration"] + 1
        ctx.log(f"best iteration {best_iters}: recall@{k}={metrics['recall_at_k']:.4f}")
    else:
        ctx.check_stop()
        S = _knn_similarity(R, int(hp.get("neighbors", 50)), float(hp.get("shrinkage", 10.0)))
        ctx.check_stop()
        metrics = _evaluate({"recipe": "item_knn", "R": R, "similarity": S}, R, test, k)
        ctx.log_metric("recall_at_k", metrics["recall_at_k"], step=0)
        ctx.log_metric("ndcg_at_k", metrics["ndcg_at_k"], step=0)
    ctx.log_metric("coverage", metrics["coverage"])
    ctx.log(f"validation: recall@{k}={metrics['recall_at_k']:.4f} ndcg@{k}={metrics['ndcg_at_k']:.4f} "
            f"coverage={metrics['coverage']:.3f} (popularity recall@{k}={base['recall_at_k']:.4f})")

    ctx.log("refitting on ALL interactions (held-out ones included) for the saved model")
    R_all = _csr(frame, nu, ni)
    model = {"recipe": recipe, "user_ids": [_native(u) for u in users], "item_ids": [_native(i) for i in items],
             "popularity": np.argsort(-np.asarray((R_all > 0).sum(axis=0)).ravel(), kind="stable"),
             "popularity_counts": np.asarray((R_all > 0).sum(axis=0)).ravel(), "k": k, "R": R_all,
             "columns": dict(zip(("user", "item", "weight", "time"), cols))}
    if recipe == "implicit_als":
        X, Y, _ = _fit_als(ctx, R_all, best_iters)
        model.update(user_factors=X, item_factors=Y)
    else:
        model["similarity"] = _knn_similarity(R_all, int(hp.get("neighbors", 50)), float(hp.get("shrinkage", 10.0)))
    out = Path(ctx.output_dir)
    joblib.dump(model, out / "model.joblib")
    (out / "io.json").write_text(json.dumps({"features": [
        {"name": "user", "dtype": "any", "required": False,
         "description": "user id to recommend for (unknown users get popular items, cold_start=true)"},
        {"name": "item", "dtype": "any", "required": False,
         "description": "instead of user: item id to find similar items for"},
        {"name": "k", "dtype": "numeric", "required": False, "description": f"number of items (default {k})"},
        {"name": "exclude_seen", "dtype": "any", "required": False,
         "description": "drop items the user already interacted with (default true)"}]}))
    metrics["popularity_recall_at_k"] = base["recall_at_k"]
    return {"metrics": metrics, "model_path": str(out / "model.joblib")}


def train(ctx) -> dict:
    return _train(ctx, "implicit_als")


def train_knn(ctx) -> dict:
    return _train(ctx, "item_knn")


# ---------------------------------------------------------------- serving

def load_model(model_dir):
    import joblib

    m = joblib.load(Path(model_dir) / "model.joblib")
    m["user_index"] = {str(u): i for i, u in enumerate(m["user_ids"])}
    m["item_index"] = {str(i): j for j, i in enumerate(m["item_ids"])}
    return m


def _flag(v, default=True):
    if v is None:
        return default
    return v if isinstance(v, bool) else str(v).lower() not in ("false", "0", "no")


def _out(m, idx, scores, **extra):
    return {"prediction": [m["item_ids"][j] for j in idx], "scores": [round(float(s), 6) for s in scores], **extra}


def predict(m, rows: list[dict]) -> list[dict]:
    import numpy as np

    results = []
    pop, counts = m["popularity"], m["popularity_counts"]
    for n, row in enumerate(rows):
        k = int(row.get("k") or m["k"])
        if row.get("user") is None and row.get("item") is None:
            raise ValueError(f"row {n}: send {{'user': <id>}} for recommendations or {{'item': <id>}} for similar items")
        if row.get("user") is None:  # similar items
            j = m["item_index"].get(str(row["item"]))
            if j is None:
                results.append(_out(m, pop[:k], counts[pop[:k]], cold_start=True))
                continue
            if m["recipe"] == "implicit_als":
                Y = m["item_factors"]
                norms = np.linalg.norm(Y, axis=1) + 1e-12
                sims = (Y @ Y[j]) / (norms * norms[j])
            else:
                sims = np.asarray(m["similarity"][j].todense()).ravel()
            top = _top(sims, k, np.array([j]))
            results.append(_out(m, top, sims[top], similar_to=row["item"]))
            continue
        u = m["user_index"].get(str(row["user"]))
        if u is None:
            results.append(_out(m, pop[:k], counts[pop[:k]], cold_start=True))
            continue
        R = m["R"]
        seen = R.indices[R.indptr[u]:R.indptr[u + 1]] if _flag(row.get("exclude_seen")) else None
        s = _scores(m, np.array([u]))[0]
        top = _top(s, k, seen)
        results.append(_out(m, top, s[top], cold_start=False))
    return results
