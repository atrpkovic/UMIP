import re
import pandas as pd

def apply_mapping(ga4_df: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    """
    Annotate GA4 rows with keyword_id using mapping rules.
    Supports pattern_type: literal | wildcard | regex
    """
    ga4 = ga4_df.copy()
    if "keyword_id" not in ga4.columns:
        ga4["keyword_id"] = None

    mapping_df = mapping_df.sort_values("map_id")
    for _, row in mapping_df.iterrows():
        pat = str(row["url_pattern"])
        ptype = str(row["pattern_type"]).lower()
        k_id = row["keyword_id"]
        if ptype == "literal":
            mask = ga4["page_path"] == pat
        elif ptype == "wildcard":
            rx = "^" + re.escape(pat).replace("\\*", ".*") + "$"
            mask = ga4["page_path"].str.match(rx, na=False)
        elif ptype == "regex":
            mask = ga4["page_path"].str.match(pat, na=False)
        else:
            continue
        ga4.loc[mask, "keyword_id"] = ga4.loc[mask, "keyword_id"].fillna(k_id)
    return ga4
