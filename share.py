import pandas as pd

def compute_share(ahrefs_df: pd.DataFrame, keyword_scope: list[str]) -> pd.DataFrame:
    """
    Estimate relative organic share by domain per keyword using Ahrefs.
    Uses 'traffic' if available; else falls back to 'search_volume'.
    """
    df = ahrefs_df[ahrefs_df["keyword"].isin(keyword_scope)].copy()
    score_col = "traffic" if "traffic" in df.columns else "search_volume"
    agg = (
        df.groupby(["keyword","domain"])[score_col]
        .sum().rename("score").reset_index()
    )
    totals = agg.groupby("keyword")["score"].sum().rename("total_score").reset_index()
    out = agg.merge(totals, on="keyword", how="left")
    out["share"] = (out["score"] / out["total_score"]).fillna(0.0)
    return out
