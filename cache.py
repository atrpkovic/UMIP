import os
import hashlib
import pandas as pd
from datetime import datetime

CACHE_FILE = "trends_cache.parquet"

def _keyhash(obj: dict) -> str:
    blob = "|".join(f"{k}={obj[k]}" for k in sorted(obj.keys()))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()

def load_cache(base_dir: str) -> pd.DataFrame:
    path = os.path.join(base_dir, CACHE_FILE)
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["key","keyword","month","trends_index"])

def save_cache(base_dir: str, df: pd.DataFrame) -> None:
    path = os.path.join(base_dir, CACHE_FILE)
    df.to_parquet(path, index=False)

def key_for(keyword: str, timeframe: str, geo: str, cat: int) -> str:
    meta = {"kw": keyword, "tf": timeframe, "geo": geo, "cat": str(cat)}
    return _keyhash(meta)

def merge_into_cache(cache: pd.DataFrame, add: pd.DataFrame) -> pd.DataFrame:
    if cache.empty:
        return add.copy()
    out = pd.concat([cache, add], ignore_index=True)
    out = out.drop_duplicates(subset=["key","keyword","month"], keep="last")
    return out

def months_covered(df: pd.DataFrame, key: str, keyword: str) -> set:
    sub = df[(df["key"] == key) & (df["keyword"] == keyword)]
    return set(pd.to_datetime(sub["month"]).dt.to_period("M").astype(str))

def month_str(dt) -> str:
    return pd.to_datetime(dt).to_period("M").strftime("%Y-%m")
