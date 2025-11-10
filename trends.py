# trends.py
import os
import time
import random
import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

# Optional SciPy (exact p-values); we fall back if missing
try:
    from scipy.stats import linregress
    SCIPY = True
except Exception:
    SCIPY = False

# Third-party
from pytrends.request import TrendReq
from pytrends import exceptions as pt_ex

# Local
from cache import load_cache, save_cache, key_for, merge_into_cache, months_covered
from config import (
    MANUAL_TRENDS_DIR,
    MAX_KEYWORDS_PER_RUN,
    TRENDS_MAX_ATTEMPTS,
)
# Optional cap (added in config patch); default to 600s if missing
try:
    from config import TRENDS_MAX_TOTAL_SECONDS
except Exception:
    TRENDS_MAX_TOTAL_SECONDS = 600  # 10 minutes per pull window

# ---------- Logging ----------
logger = logging.getLogger("umip.trends")
if not logger.handlers:
    # If the app didn't configure logging, set a basic one so users see progress.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

# ---------- Fast-fail on 429 ----------
class Fast429(Exception):
    """Signal to bail immediately on a 429 and use fallback."""
    pass

FAST_FAIL_ON_429 = bool(int(os.environ.get("UMIP_FAST_FAIL_ON_429", "1")))  # 1 = enabled

# ---------- TrendReq factory (keep simple; we do our own backoff) ----------
def _new_trendreq() -> TrendReq:
    return TrendReq(
        hl="en-US",
        tz=360,
        timeout=(15, 60),
        requests_args={
            "headers": {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/123.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            }
        },
    )

# ---------- Stats helpers ----------
@dataclass
class TrendStats:
    slope: float
    intercept: float
    r_value: float
    p_value: float
    r_squared: float
    n: int

def month_index_series(df: pd.DataFrame) -> pd.DataFrame:
    s = df.sort_values("month").copy()
    s["t"] = np.arange(1, len(s) + 1)
    return s

def linear_trend(df_kw: pd.DataFrame) -> TrendStats:
    s = month_index_series(df_kw)
    x = s["t"].values.astype(float)
    y = s["trends_index"].values.astype(float)
    if SCIPY:
        r = linregress(x, y)
        return TrendStats(
            slope=r.slope,
            intercept=r.intercept,
            r_value=r.rvalue,
            p_value=r.pvalue,
            r_squared=r.rvalue ** 2,
            n=len(x),
        )
    # Fallback approximate p-value
    r = np.corrcoef(x, y)[0, 1]
    n = len(x)
    slope = r * (y.std(ddof=1) / x.std(ddof=1))
    intercept = y.mean() - slope * x.mean()
    t_stat = r * np.sqrt((n - 2) / (1 - r**2 + 1e-9))
    from math import erf, sqrt
    p_approx = 2.0 * (1 - 0.5 * (1 + erf(abs(t_stat) / np.sqrt(2))))
    return TrendStats(slope, intercept, r, p_approx, r**2, n)

def seasonal_indices(df_kw: pd.DataFrame) -> pd.Series:
    s = df_kw.copy()
    s["month_num"] = pd.to_datetime(s["month"]).dt.month
    monthly_avg = s.groupby("month_num")["trends_index"].mean()
    overall_avg = s["trends_index"].mean()
    return monthly_avg / overall_avg

def forecast_next_quarter(df_kw: pd.DataFrame) -> Tuple[pd.DataFrame, TrendStats]:
    s = month_index_series(df_kw)
    ts = linear_trend(df_kw)
    seas = seasonal_indices(df_kw)
    last_t = s["t"].max()
    last_month = pd.to_datetime(s["month"].max())
    hist = s["trends_index"].values
    p90 = np.percentile(hist, 90)

    rows = []
    for i in range(1, 4):
        t_i = last_t + i
        month_i = (last_month + relativedelta(months=+i)).to_period("M").to_timestamp()
        base = ts.intercept + ts.slope * t_i
        mnum = int(pd.to_datetime(month_i).month)
        seas_mult = float(seas.get(mnum, 1.0))
        yhat = max(0.0, base * seas_mult)
        rows.append({"month": month_i, "forecast_index": yhat})
    fdf = pd.DataFrame(rows)
    fdf["is_peak_flag"] = fdf["forecast_index"] >= p90
    return fdf, ts

# ---------- PyTrends helpers ----------
def _normalize_with_anchor(chunk_df: pd.DataFrame, anchor: str, master_anchor_mean: float) -> pd.DataFrame:
    """
    Scale all series in a chunk so that the chunk's anchor mean matches the master anchor mean.
    Expects columns: keyword, month, trends_index
    """
    df = chunk_df.copy()
    anchor_mean = df.loc[df["keyword"] == anchor, "trends_index"].replace(0, np.nan).dropna().mean()
    if not np.isfinite(anchor_mean) or anchor_mean == 0:
        return df
    scale = master_anchor_mean / anchor_mean
    df["trends_index"] = (df["trends_index"] * scale).clip(lower=0)
    return df

def _iot_to_long(iot: pd.DataFrame) -> pd.DataFrame:
    df = iot.drop(columns=[c for c in iot.columns if c.lower() == "ispartial"], errors="ignore").copy()
    df = df.reset_index().melt(id_vars="date", var_name="keyword", value_name="trends_index")
    df = df.rename(columns={"date": "month"})
    return df

def _pull_iot_with_retry(
    pytrends: TrendReq,
    terms: List[str],
    *,
    timeframe: str,
    geo: str,
    cat: int,
    max_attempts: int,
    base_sleep: float,
    max_sleep: float,
):
    start_ts = time.time()
    attempt = 0
    while True:
        attempt += 1
        try:
            logger.info(f"[pull] terms={terms} tf='{timeframe}' geo={geo} cat={cat} attempt={attempt}/{max_attempts}")
            pytrends.build_payload(terms, timeframe=timeframe, geo=geo, cat=cat)
            iot = pytrends.interest_over_time()
            if iot is None or iot.empty:
                raise RuntimeError("Empty interest_over_time response")
            logger.info(f"[pull] SUCCESS terms={terms} rows={len(iot)}")
            return iot
        except (pt_ex.TooManyRequestsError, pt_ex.ResponseError, RuntimeError) as e:
            elapsed = int(time.time() - start_ts)

            # Fast-fail on 429 to trigger manual CSV fallback immediately
            if FAST_FAIL_ON_429 and isinstance(e, pt_ex.TooManyRequestsError):
                logger.warning("[pull] 429 fast-fail enabled; bailing to fallback immediately")
                raise Fast429("google trends 429")

            if attempt >= max_attempts:
                logger.error(f"[pull] FAIL after {attempt} attempts ({elapsed}s): {e}")
                raise
            if elapsed >= TRENDS_MAX_TOTAL_SECONDS:
                logger.error(f"[pull] TIMEOUT after {elapsed}s (cap={TRENDS_MAX_TOTAL_SECONDS})")
                raise
            sleep_s = min(max_sleep, base_sleep * (2 ** (attempt - 1))) + random.uniform(0, 2.5)
            logger.warning(f"[pull] 429/err. Sleeping {sleep_s:.1f}s (elapsed={elapsed}s)…")
            time.sleep(sleep_s)

# ---------- Manual CSV fallback (exported from trends.google.com) ----------
def _load_manual_trends(keyword: str) -> pd.DataFrame | None:
    """
    Accept a CSV exported from trends.google.com (Monthly).
    Expected columns often look like:
      'Month','<keyword>: (United States)' OR 'Date','<keyword>'
    We normalize to: keyword, month, trends_index
    """
    safe_name = f"{keyword}.csv"
    path = os.path.join(MANUAL_TRENDS_DIR, safe_name)
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)

    # Detect date column
    date_col = None
    for c in df.columns:
        if str(c).lower() in ("month", "date"):
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]  # fallback

    # Detect value column (first numeric-ish / keyword-ish column)
    val_col = None
    for c in df.columns:
        if c == date_col:
            continue
        if df[c].dtype.kind in "biufc" or "value" in str(c).lower() or keyword.lower() in str(c).lower():
            val_col = c
            break
    if val_col is None:
        if len(df.columns) >= 2:
            val_col = df.columns[1]
        else:
            return None

    out = pd.DataFrame(
        {
            "keyword": keyword,
            "month": pd.to_datetime(df[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp(),
            "trends_index": pd.to_numeric(df[val_col], errors="coerce").fillna(0.0),
        }
    ).dropna(subset=["month"])

    return out

# ---------- Main fetcher (pytrends only, with manual CSV fallback) ----------
def fetch_trends_timeseries(
    keywords: list[str],
    base_dir: str,
    timeframe: str = "today 5-y",
    geo: str = "US",
    cat: int = 0,
    anchor: str = "tires",
    sleep_min: int = 10,  # gentle defaults
    sleep_max: int = 20,
) -> pd.DataFrame:
    """
    Super-gentle pytrends fetch with:
      - CHUNK=1 (+anchor)
      - Long exponential backoff + fast-fail on 429
      - Manual CSV fallback per keyword (from MANUAL_TRENDS_DIR)
      - Hard skip on persistent failure (logs to console) so pipeline continues
      - MAX_KEYWORDS_PER_RUN to warm cache gradually
    Returns long df: keyword, month, trends_index
    """
    cache = load_cache(base_dir)
    pytrends = _new_trendreq()

    # Respect max keywords per run (but always include anchor if present)
    kws = [k for k in keywords if isinstance(k, str) and k.strip()]
    if anchor in kws:
        kws = [anchor] + [k for k in kws if k != anchor]
    if MAX_KEYWORDS_PER_RUN > 0:
        kws = kws[: max(1, MAX_KEYWORDS_PER_RUN + (1 if anchor in kws else 0))]

    logger.info(f"[fetch] starting; keywords={len(kws)} (cap per run={MAX_KEYWORDS_PER_RUN}) anchor='{anchor}'")
    out_long: List[pd.DataFrame] = []

    # ---- Ensure anchor series exists (or load manual) ----
    logger.info("[fetch] ensuring anchor series present")
    anchor_key = key_for(anchor, timeframe, geo, cat)
    covered_anchor = months_covered(cache, anchor_key, anchor)

    if len(covered_anchor) < 55:
        try:
            iot = _pull_iot_with_retry(
                pytrends,
                [anchor],
                timeframe=timeframe,
                geo=geo,
                cat=cat,
                max_attempts=TRENDS_MAX_ATTEMPTS,
                base_sleep=float(sleep_min),
                max_sleep=max(60.0, float(sleep_max) * 6.0),
            )
            base_long = _iot_to_long(iot)
        except Fast429:
            manual = _load_manual_trends(anchor)
            if manual is None or manual.empty:
                logger.error(f"[fetch] Anchor '{anchor}' 429 and no manual CSV. Skipping Trends entirely.")
                return pd.DataFrame(columns=["keyword", "month", "trends_index"])
            logger.warning(f"[manual] using manual CSV for anchor '{anchor}' from {MANUAL_TRENDS_DIR}")
            base_long = manual
        except Exception:
            manual = _load_manual_trends(anchor)
            if manual is None or manual.empty:
                logger.error(f"[fetch] Anchor '{anchor}' failed and no manual CSV. Skipping Trends entirely.")
                return pd.DataFrame(columns=["keyword", "month", "trends_index"])
            logger.warning(f"[manual] using manual CSV for anchor '{anchor}' from {MANUAL_TRENDS_DIR}")
            base_long = manual
        add = base_long.copy()
        add.insert(0, "key", anchor_key)
        cache = merge_into_cache(cache, add[["key", "keyword", "month", "trends_index"]])
        save_cache(base_dir, cache)
        time.sleep(random.uniform(sleep_min, sleep_max))
    else:
        base_long = cache[
            (cache["key"] == anchor_key) & (cache["keyword"] == anchor)
        ][["keyword", "month", "trends_index"]].copy()

    logger.info(f"[fetch] anchor ready; months={base_long['month'].nunique()}")

    # ---- Anchor baseline for normalization ----
    master_anchor_mean = (
        base_long.loc[base_long["keyword"] == anchor, "trends_index"].replace(0, np.nan).dropna().mean()
    )
    if not np.isfinite(master_anchor_mean) or master_anchor_mean == 0:
        master_anchor_mean = max(
            1.0, base_long["trends_index"].replace(0, np.nan).dropna().mean() or 1.0
        )

    out_long.append(base_long)

    # ---- Fetch remaining keywords one-by-one with anchor ----
    others = [k for k in kws if k != anchor]
    for k in others:
        logger.info(f"[fetch] keyword='{k}'")
        terms = [k, anchor]  # CHUNK = 1 (+ anchor)
        kkey = key_for(k, timeframe, geo, cat)
        covered = months_covered(cache, kkey, k)

        if len(covered) >= 55:
            # read from cache
            sub = cache[(cache["key"] == kkey) & (cache["keyword"] == k)][
                ["keyword", "month", "trends_index"]
            ].copy()
            logger.info(f"[cache] hit for '{k}' months={sub['month'].nunique()}")
            if not sub.empty:
                out_long.append(sub)
            continue

        # try pytrends pull for this single keyword (+anchor)
        try:
            iot = _pull_iot_with_retry(
                pytrends,
                terms,
                timeframe=timeframe,
                geo=geo,
                cat=cat,
                max_attempts=TRENDS_MAX_ATTEMPTS,
                base_sleep=float(sleep_min),
                max_sleep=max(60.0, float(sleep_max) * 6.0),
            )
            chunk_long = _iot_to_long(iot)
        except Fast429:
            manual = _load_manual_trends(k)
            if manual is None or manual.empty:
                logger.error(f"[skip] '{k}' 429 and no manual CSV")
                continue
            logger.warning(f"[manual] using manual CSV for '{k}' from {MANUAL_TRENDS_DIR}")
            # merge manual with current anchor (for normalization)
            chunk_long = pd.concat([manual, base_long], ignore_index=True)
        except Exception:
            manual = _load_manual_trends(k)
            if manual is None or manual.empty:
                logger.error(f"[skip] no data for '{k}' (pytrends error and no manual CSV)")
                continue
            logger.warning(f"[manual] using manual CSV for '{k}' from {MANUAL_TRENDS_DIR}")
            # merge manual with current anchor (for normalization)
            chunk_long = pd.concat([manual, base_long], ignore_index=True)

        # normalize to anchor baseline
        chunk_norm = _normalize_with_anchor(chunk_long, anchor, master_anchor_mean)

        # append only this keyword’s normalized series & write to cache
        out_long.append(chunk_norm[chunk_norm["keyword"] == k])

        sub = chunk_norm[chunk_norm["keyword"] == k].copy()
        sub.insert(0, "key", kkey)
        cache = merge_into_cache(cache, sub[["key", "keyword", "month", "trends_index"]])
        save_cache(base_dir, cache)

        time.sleep(random.uniform(sleep_min, sleep_max))

    ts = pd.concat(out_long, ignore_index=True)
    ts["month"] = pd.to_datetime(ts["month"]).dt.to_period("M").dt.to_timestamp()
    ts = ts.drop_duplicates(subset=["keyword", "month"], keep="last").sort_values(["keyword", "month"]).reset_index(drop=True)
    logger.info(f"[fetch] done. total keywords={ts['keyword'].nunique()} total rows={len(ts)}")
    return ts
