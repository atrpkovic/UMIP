import os
from dotenv import load_dotenv

load_dotenv()

def env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)

def env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except Exception:
        return default

BASE_DIR = env_str("UMIP_DATA_DIR", "./data")
OUTPUT_XLSX = env_str("UMIP_OUTPUT_XLSX", "umip_quarterly_report.xlsx")

# Trends config
TRENDS_GEO = env_str("UMIP_TRENDS_GEO", "US")
TRENDS_CATEGORY = env_int("UMIP_TRENDS_CATEGORY", 0)
TRENDS_ANCHOR = env_str("UMIP_TRENDS_ANCHOR", "tires")
TRENDS_SLEEP_MIN = env_int("UMIP_TRENDS_SLEEP_MIN", 2)
TRENDS_SLEEP_MAX = env_int("UMIP_TRENDS_SLEEP_MAX", 6)

# CSV filenames (relative to BASE_DIR)
KW_MASTER_CSV = "keywords_master.csv"
KW_MAPPING_CSV = "keyword_mapping.csv"
AHREFS_EXPORT_CSV = "ahrefs_keywords_export.csv"
TRENDS_TS_CSV = "trends_timeseries.csv"     # output of fetcher
GA4_METRICS_CSV = "ga4_page_metrics.csv"
# ----- add at the end of config.py -----
# Optional folder with manually-downloaded Google Trends CSVs.
# Put files like: data/trends_manual/winter tires.csv  (one keyword per file)
MANUAL_TRENDS_DIR = env_str("UMIP_MANUAL_TRENDS_DIR", "./data/trends_manual")

# Limit how many keywords to fetch per run (helps warm cache without 429s)
MAX_KEYWORDS_PER_RUN = env_int("UMIP_MAX_KEYWORDS_PER_RUN", 3)

# Backoff / attempts (tune if needed)
TRENDS_MAX_ATTEMPTS = env_int("UMIP_TRENDS_MAX_ATTEMPTS", 8)
# --- Logging & retry caps ---
import os

LOG_LEVEL = os.environ.get("UMIP_LOG_LEVEL", "INFO")  # DEBUG|INFO|WARNING|ERROR
TRENDS_MAX_ATTEMPTS = env_int("UMIP_TRENDS_MAX_ATTEMPTS", 8)  # already referenced in trends.py
TRENDS_MAX_TOTAL_SECONDS = env_int("UMIP_TRENDS_MAX_TOTAL_SECONDS", 600)  # 10 min safety for a single run
