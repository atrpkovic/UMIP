# report.py
import os
import logging
from typing import List, Tuple

import pandas as pd
import numpy as np

from trends import fetch_trends_timeseries, forecast_next_quarter, linear_trend

# ---- Config imports (with safe fallbacks so this file works even if some keys are missing) ----
try:
    from config import (
        BASE_DIR,
        KW_MASTER_CSV,
        OUTPUT_XLSX,
        TRENDS_GEO,
        TRENDS_CATEGORY,
        TRENDS_ANCHOR,
        TRENDS_SLEEP_MIN,
        TRENDS_SLEEP_MAX,
        MAX_KEYWORDS_PER_RUN,
    )
except Exception:
    # Sensible defaults if config keys aren’t present
    BASE_DIR = "."
    KW_MASTER_CSV = "./data/keywords_master.csv"
    OUTPUT_XLSX = "./data/umip_quarterly_report.xlsx"
    TRENDS_GEO = "US"
    TRENDS_CATEGORY = 0
    TRENDS_ANCHOR = "tires"
    TRENDS_SLEEP_MIN = 10
    TRENDS_SLEEP_MAX = 20
    MAX_KEYWORDS_PER_RUN = 3

# Optional logging level from config
try:
    from config import LOG_LEVEL
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
except Exception:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

logger = logging.getLogger("umip.report")


def _p(path: str) -> str:
    """Resolve a path relative to the runtime CWD."""
    return os.path.abspath(path)


def _read_keywords(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Keywords CSV not found: {path}")
    df = pd.read_csv(path)
    # normalize expected columns
    # allow either 'keyword' or 'keywords' or first col fallback
    if "keyword" not in df.columns:
        if "keywords" in df.columns:
            df = df.rename(columns={"keywords": "keyword"})
        else:
            first_col = df.columns[0]
            df = df.rename(columns={first_col: "keyword"})
    # optional columns we keep if present (type, sv, cpc, etc., from Ahrefs export)
    keep_cols = [c for c in ["keyword", "type", "sv", "cpc"] if c in df.columns]
    df = df[keep_cols] if keep_cols else df[["keyword"]]
    # trim whitespace and drop blanks
    df["keyword"] = df["keyword"].astype(str).str.strip()
    df = df[df["keyword"] != ""].drop_duplicates(subset=["keyword"]).reset_index(drop=True)
    return df


def _compute_trend_rows(ts_long: pd.DataFrame, keywords: List[str]) -> pd.DataFrame:
    """
    ts_long: long DF with columns [keyword, month, trends_index]
    returns one row per keyword with stats + next-quarter forecast and reliability flag
    """
    out_rows = []
    for kw in keywords:
        sub = ts_long[ts_long["keyword"] == kw].dropna(subset=["trends_index"]).copy()
        if sub.empty:
            out_rows.append(
                {
                    "keyword": kw,
                    "n_months": 0,
                    "slope": np.nan,
                    "r_squared": np.nan,
                    "p_value": np.nan,
                    "reliable": False,
                    "fcast_m1": np.nan,
                    "fcast_m2": np.nan,
                    "fcast_m3": np.nan,
                    "peak_m1": False,
                    "peak_m2": False,
                    "peak_m3": False,
                }
            )
            continue

        # guardrail: require at least some non-zero values
        nz = (sub["trends_index"].fillna(0) > 0).sum()
        n = len(sub)

        try:
            fdf, stats = forecast_next_quarter(sub)
            # Unpack forecast months (sorted order)
            fdf = fdf.sort_values("month").reset_index(drop=True)
            fcasts = fdf["forecast_index"].tolist()
            peaks = fdf["is_peak_flag"].tolist()
        except Exception:
            stats = linear_trend(sub)  # at least provide slope/r/p
            fcasts, peaks = [np.nan, np.nan, np.nan], [False, False, False]

        # Reliability rule (adjust thresholds as needed):
        # - at least 36 months of data
        # - p < 0.05
        # You can also add a floor for R^2 if desired (e.g., >= 0.15)
        reliable = (stats.n >= 36) and (stats.p_value is not None) and (stats.p_value < 0.05)

        out_rows.append(
            {
                "keyword": kw,
                "n_months": int(n),
                "slope": float(stats.slope) if stats.slope is not None else np.nan,
                "r_squared": float(stats.r_squared) if stats.r_squared is not None else np.nan,
                "p_value": float(stats.p_value) if stats.p_value is not None else np.nan,
                "reliable": bool(reliable),
                "fcast_m1": float(fcasts[0]) if len(fcasts) > 0 else np.nan,
                "fcast_m2": float(fcasts[1]) if len(fcasts) > 1 else np.nan,
                "fcast_m3": float(fcasts[2]) if len(fcasts) > 2 else np.nan,
                "peak_m1": bool(peaks[0]) if len(peaks) > 0 else False,
                "peak_m2": bool(peaks[1]) if len(peaks) > 1 else False,
                "peak_m3": bool(peaks[2]) if len(peaks) > 2 else False,
            }
        )
    return pd.DataFrame(out_rows)


def build_quarterly_report() -> str:
    logger.info("Starting quarterly report…")

    # -----------------------
    # 1) Load keywords (from your Ahrefs-curated list)
    # -----------------------
    kw_path = _p(KW_MASTER_CSV)
    keywords_df = _read_keywords(kw_path)
    keywords = keywords_df["keyword"].tolist()
    logger.info(f"[report] keywords loaded: {len(keywords)} from {kw_path}")

    # Respect MAX_KEYWORDS_PER_RUN if set (>0)
    if MAX_KEYWORDS_PER_RUN and MAX_KEYWORDS_PER_RUN > 0:
        keywords = keywords[:MAX_KEYWORDS_PER_RUN]
        logger.info(f"[report] capped to MAX_KEYWORDS_PER_RUN={MAX_KEYWORDS_PER_RUN}: now {len(keywords)} keywords")

    # -----------------------
    # 2) Pull Google Trends (normalized with anchor)
    # -----------------------
    timeframe = "today 5-y"
    ts_long = fetch_trends_timeseries(
        keywords=keywords,
        base_dir=BASE_DIR,
        timeframe=timeframe,
        geo=TRENDS_GEO,
        cat=TRENDS_CATEGORY,
        anchor=TRENDS_ANCHOR,
        sleep_min=TRENDS_SLEEP_MIN,
        sleep_max=TRENDS_SLEEP_MAX,
    )
    if ts_long.empty:
        logger.warning("[report] trends timeseries came back empty")
    else:
        logger.info(f"[report] trends rows: {len(ts_long)} for {ts_long['keyword'].nunique()} keywords")

    # -----------------------
    # 3) Compute trend stats & next-quarter forecast + reliability
    # -----------------------
    summary_df = _compute_trend_rows(ts_long, keywords)

    # Bring through any optional fields from the keywords file (e.g., type, sv, cpc)
    if not keywords_df.empty:
        # Avoid duplicating the 'keyword' column
        merge_cols = [c for c in keywords_df.columns if c != "keyword"]
        if merge_cols:
            summary_df = summary_df.merge(keywords_df[["keyword"] + merge_cols], on="keyword", how="left")

    # -----------------------
    # 4) (COMMENTED OUT) Ahrefs competitor context
    # -----------------------
    # from utils import load_ahrefs_competitor_exports, compute_share_of_traffic
    # ahrefs_df = load_ahrefs_competitor_exports()  # expects your manual CSV exports
    # comp_df = compute_share_of_traffic(ahrefs_df) # your function to estimate share vs. competitors
    # # Join into summary if desired
    # summary_df = summary_df.merge(comp_df, on="keyword", how="left")

    # -----------------------
    # 5) (COMMENTED OUT) GA4 ingestion & mapping
    # -----------------------
    # from utils import load_ga4_combined, apply_keyword_page_mapping
    # ga4_df = load_ga4_combined()  # reads GA4 CSVs defined in config
    # mapped_ga4 = apply_keyword_page_mapping(ga4_df, mapping_rules=...)  # your mapping rules/table
    # # Aggregate to keyword-level as needed and join:
    # kpi_df = mapped_ga4.groupby("keyword", as_index=False).agg(
    #     sessions=("sessions", "sum"),
    #     key_events=("key_events", "sum"),
    #     revenue=("revenue", "sum"),
    # )
    # summary_df = summary_df.merge(kpi_df, on="keyword", how="left")

    # -----------------------
    # 6) Write Excel (Trends-only for now)
    # -----------------------
    out_path = _p(OUTPUT_XLSX)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as xw:
        # Sheet 1: Keyword summary (with reliability flags)
        summary_order = [
            "keyword",
            # optional info if present:
            *([c for c in ["type", "sv", "cpc"] if c in summary_df.columns]),
            "n_months",
            "slope",
            "r_squared",
            "p_value",
            "reliable",
            "fcast_m1",
            "fcast_m2",
            "fcast_m3",
            "peak_m1",
            "peak_m2",
            "peak_m3",
        ]
        existing_cols = [c for c in summary_order if c in summary_df.columns]
        summary_df[existing_cols].to_excel(xw, sheet_name="Keywords", index=False)

        # Add basic conditional formatting for reliability (green/red)
        try:
            wb = xw.book
            ws = xw.sheets["Keywords"]
            yes_fmt = wb.add_format({"bg_color": "#C6EFCE", "font_color": "#006100"})
            no_fmt = wb.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006"})
            # Find 'reliable' column
            if "reliable" in existing_cols:
                col_idx = existing_cols.index("reliable")
                # Excel range: rows start at 2 (1-based; header in row 1)
                last_row = len(summary_df) + 1
                col_letter = chr(ord('A') + col_idx)
                rng = f"{col_letter}2:{col_letter}{last_row}"
                ws.conditional_format(rng, {"type": "cell", "criteria": "==", "value": True, "format": yes_fmt})
                ws.conditional_format(rng, {"type": "cell", "criteria": "==", "value": False, "format": no_fmt})
        except Exception:
            # best-effort; ignore formatting errors
            pass

        # Sheet 2: Full normalized time series
        if not ts_long.empty:
            ts_long.sort_values(["keyword", "month"]).to_excel(xw, sheet_name="Trends_TS", index=False)

        # (COMMENTED OUT) Additional sheets for GA4/Ahrefs can be added later
        # if not ga4_df.empty:
        #     ga4_df.to_excel(xw, sheet_name="GA4", index=False)
        # if not comp_df.empty:
        #     comp_df.to_excel(xw, sheet_name="Competitors", index=False)

    logger.info(f"Report written to: {out_path}")
    return out_path
