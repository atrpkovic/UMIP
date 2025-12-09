#!/usr/bin/env python3
"""
UMIP - Universal Marketing Intelligence Platform
Snowflake-integrated version for centralized data storage.
"""

import os
import json
import hashlib
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from pathlib import Path
from dotenv import load_dotenv
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
CACHE_DIR = Path("cache")
OUTPUT_DIR = Path("output")
SERPAPI_BASE_URL = "https://serpapi.com/search"
P_VALUE_THRESHOLD = 0.05


# ============================================
# Snowflake Connection
# ============================================

def get_snowflake_connection():
    """Create Snowflake connection from environment variables."""
    load_dotenv()
    
    required_vars = [
        "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_USER", 
        "SNOWFLAKE_PASSWORD",
        "SNOWFLAKE_WAREHOUSE"
    ]
    
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        raise ValueError(f"Missing Snowflake environment variables: {missing}")
    
    return snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE", "PRIORITY_TIRE_DATA"),
        schema=os.getenv("SNOWFLAKE_SCHEMA", "UMIP")
    )


def execute_query(query: str, params: tuple = None) -> pd.DataFrame:
    """Execute a query and return results as DataFrame."""
    conn = get_snowflake_connection()
    try:
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        return pd.DataFrame(rows, columns=columns)
    finally:
        conn.close()


def execute_write(query: str, params: tuple = None):
    """Execute a write query (INSERT, UPDATE, DELETE)."""
    conn = get_snowflake_connection()
    try:
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        conn.commit()
    finally:
        conn.close()


# ============================================
# Data Loading - From Snowflake
# ============================================

def load_keywords_from_snowflake() -> list:
    """Load keywords from Snowflake."""
    query = """
        SELECT KEYWORD 
        FROM PRIORITY_TIRE_DATA.UMIP.AHREFS_KEYWORDS 
        ORDER BY KEYWORD
    """
    df = execute_query(query)
    if df.empty:
        return []
    return df["KEYWORD"].tolist()


def load_ahrefs_data_from_snowflake() -> pd.DataFrame:
    """Load Ahrefs keyword data from Snowflake."""
    query = """
        SELECT 
            KEYWORD,
            SEARCH_VOLUME,
            CPC,
            KD,
            TRAFFIC_POTENTIAL,
            EXPORT_DATE
        FROM PRIORITY_TIRE_DATA.UMIP.AHREFS_KEYWORDS
        ORDER BY KEYWORD
    """
    return execute_query(query)


def load_ga4_data_from_snowflake() -> pd.DataFrame:
    """Load GA4 page metrics from Snowflake."""
    query = """
        SELECT 
            MONTH,
            PAGE_PATH,
            PAGE_TYPE,
            SESSIONS,
            KEY_EVENTS,
            TRANSACTIONS,
            REVENUE,
            AVAILABLE_QTY
        FROM PRIORITY_TIRE_DATA.UMIP.GA4_PAGE_METRICS
        ORDER BY MONTH DESC
    """
    return execute_query(query)


# ============================================
# Data Upload - To Snowflake
# ============================================

def upload_keywords_to_snowflake(keywords: list, category: str = None):
    """Upload keywords to Snowflake master list."""
    conn = get_snowflake_connection()
    try:
        cursor = conn.cursor()
        
        for kw in keywords:
            cursor.execute("""
                MERGE INTO PRIORITY_TIRE_DATA.UMIP.KEYWORDS_MASTER AS target
                USING (SELECT %s AS KEYWORD, %s AS CATEGORY) AS source
                ON target.KEYWORD = source.KEYWORD
                WHEN NOT MATCHED THEN
                    INSERT (KEYWORD, CATEGORY, IS_ACTIVE)
                    VALUES (source.KEYWORD, source.CATEGORY, TRUE)
            """, (kw, category))
        
        conn.commit()
        print(f"Uploaded {len(keywords)} keywords to Snowflake")
    finally:
        conn.close()


def upload_ahrefs_csv_to_snowflake(csv_path: str):
    """Upload Ahrefs CSV export to Snowflake (replaces existing data)."""
    # Ahrefs exports as UTF-16 with tab separator
    try:
        df = pd.read_csv(csv_path, encoding="utf-16", sep="\t")
    except:
        # Fallback to UTF-8 if UTF-16 fails
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    
    # Map Ahrefs columns to our schema
    upload_df = pd.DataFrame()
    upload_df["KEYWORD"] = df["Keyword"]
    upload_df["CURRENT_URL"] = df.get("Current URL", None)
    upload_df["PAGE_PATH"] = df["Current URL"].apply(
        lambda x: x.replace("https://www.prioritytire.com", "").replace("×", "x") if pd.notna(x) else None
    ) if "Current URL" in df.columns else None
    upload_df["VOLUME"] = df.get("Volume", 0)
    upload_df["CURRENT_POSITION"] = df.get("Current position", None)
    upload_df["CURRENT_TRAFFIC"] = df.get("Current traffic", 0)
    upload_df["KEYWORD_DIFFICULTY"] = df.get("Keyword Difficulty", 0)
    upload_df["CPC"] = df.get("Cost per click", 0)
    
    conn = get_snowflake_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("TRUNCATE TABLE PRIORITY_TIRE_DATA.UMIP.AHREFS_KEYWORDS")
        
        success, nchunks, nrows, _ = write_pandas(
            conn, 
            upload_df, 
            "AHREFS_KEYWORDS",
            database="PRIORITY_TIRE_DATA",
            schema="UMIP"
        )
        
        print(f"Uploaded {nrows} Ahrefs keyword records to Snowflake")
    finally:
        conn.close()


def upload_ga4_csv_to_snowflake(csv_path: str):
    """Upload GA4 CSV export to Snowflake (replaces existing data)."""
    # GA4 exports have 6 header rows to skip
    df = pd.read_csv(csv_path, skiprows=6)
    
    # Clean up the data - remove grand total row and empty landing pages
    df = df[df["Landing page"].notna()]
    df = df[df["Landing page"] != "(not set)"]
    
    # Map GA4 columns to our schema
    upload_df = pd.DataFrame()
    upload_df["PAGE_PATH"] = df["Landing page"]
    upload_df["TOTAL_USERS"] = pd.to_numeric(df["Total users"], errors="coerce").fillna(0).astype(int)
    upload_df["USER_KEY_EVENT_RATE"] = pd.to_numeric(df["User key event rate"], errors="coerce").fillna(0)
    upload_df["TOTAL_REVENUE"] = pd.to_numeric(df["Total revenue"], errors="coerce").fillna(0)
    
    # Reset index to avoid write_pandas warning
    upload_df = upload_df.reset_index(drop=True)
    
    conn = get_snowflake_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("TRUNCATE TABLE PRIORITY_TIRE_DATA.UMIP.GA4_PAGE_METRICS")
        
        success, nchunks, nrows, _ = write_pandas(
            conn,
            upload_df,
            "GA4_PAGE_METRICS",
            database="PRIORITY_TIRE_DATA",
            schema="UMIP"
        )
        
        print(f"Uploaded {nrows} GA4 page records to Snowflake")
    finally:
        conn.close()


def save_trends_to_snowflake(keyword: str, df: pd.DataFrame):
    """Save Google Trends time series to Snowflake."""
    if df.empty:
        return
    
    conn = get_snowflake_connection()
    try:
        cursor = conn.cursor()
        
        # Delete existing data for this keyword
        cursor.execute(
            "DELETE FROM PRIORITY_TIRE_DATA.UMIP.GOOGLE_TRENDS_TIMESERIES WHERE KEYWORD = %s",
            (keyword,)
        )
        
        # Prepare data
        trends_df = df.copy()
        trends_df["KEYWORD"] = keyword
        trends_df["TREND_DATE"] = trends_df["date"].dt.date
        trends_df["INTEREST"] = trends_df["interest"]
        trends_df["WEEK_NUM"] = trends_df["week_num"]
        trends_df["FETCHED_AT"] = datetime.now()
        
        upload_df = trends_df[["KEYWORD", "TREND_DATE", "INTEREST", "WEEK_NUM", "FETCHED_AT"]]
        
        success, nchunks, nrows, _ = write_pandas(
            conn,
            upload_df,
            "GOOGLE_TRENDS_TIMESERIES",
            database="PRIORITY_TIRE_DATA",
            schema="UMIP"
        )
        
        conn.commit()
    finally:
        conn.close()


def save_analysis_to_snowflake(results: list):
    """Save all analysis results to Snowflake analytics tables."""
    conn = get_snowflake_connection()
    analysis_date = datetime.now().date()
    
    try:
        cursor = conn.cursor()
        
        for r in results:
            keyword = r["keyword"]
            tm = r["trend_metrics"]
            sq = r["seasonality"]
            sm = r["monthly_seasonality"]
            data_points = len(r["raw_data"]) if not r["raw_data"].empty else 0
            
            # Insert trend analysis
            cursor.execute("""
                INSERT INTO PRIORITY_TIRE_DATA.UMIP.TREND_ANALYSIS 
                (KEYWORD, ANALYSIS_DATE, SLOPE, P_VALUE, ANNUAL_GROWTH, 
                 TREND_CLASSIFICATION, DATA_POINTS)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                keyword, analysis_date, tm["slope"], tm["p_value"],
                tm["annual_growth"], tm["trend"], data_points
            ))
            
            # Insert quarterly seasonality
            cursor.execute("""
                INSERT INTO PRIORITY_TIRE_DATA.UMIP.SEASONALITY_QUARTERLY
                (KEYWORD, ANALYSIS_DATE, Q1_AVG, Q2_AVG, Q3_AVG, Q4_AVG, PEAK_QUARTER)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                keyword, analysis_date, sq["Q1"], sq["Q2"], sq["Q3"], sq["Q4"],
                sq["peak_quarter"]
            ))
            
            # Insert monthly seasonality
            cursor.execute("""
                INSERT INTO PRIORITY_TIRE_DATA.UMIP.SEASONALITY_MONTHLY
                (KEYWORD, ANALYSIS_DATE, JAN_AVG, FEB_AVG, MAR_AVG, APR_AVG,
                 MAY_AVG, JUN_AVG, JUL_AVG, AUG_AVG, SEP_AVG, OCT_AVG,
                 NOV_AVG, DEC_AVG, PEAK_MONTH)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                keyword, analysis_date,
                sm["Jan"], sm["Feb"], sm["Mar"], sm["Apr"],
                sm["May"], sm["Jun"], sm["Jul"], sm["Aug"],
                sm["Sep"], sm["Oct"], sm["Nov"], sm["Dec"],
                sm["peak_month"]
            ))
        
        conn.commit()
        print(f"Saved analysis results for {len(results)} keywords to Snowflake")
    finally:
        conn.close()


# ============================================
# Google Trends Fetching (unchanged from original)
# ============================================

def load_config():
    """Load environment variables."""
    load_dotenv()
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        raise ValueError("SERPAPI_KEY not found in environment")
    return {"serpapi_key": api_key}


def get_cache_path(keyword: str) -> Path:
    """Generate cache file path for a keyword."""
    CACHE_DIR.mkdir(exist_ok=True)
    hash_key = hashlib.md5(keyword.lower().encode()).hexdigest()[:12]
    return CACHE_DIR / f"trends_{hash_key}.json"


def is_cached(keyword: str) -> bool:
    """Check if keyword has valid cached data."""
    cache_path = get_cache_path(keyword)
    if cache_path.exists():
        cache_age = datetime.now().timestamp() - cache_path.stat().st_mtime
        if cache_age < 86400 * 7:  # Valid for 7 days
            return True
    return False


def fetch_google_trends(keyword: str, api_key: str, use_cache: bool = True) -> dict:
    """Fetch Google Trends data via SerpAPI."""
    cache_path = get_cache_path(keyword)
    
    if use_cache and cache_path.exists():
        cache_age = datetime.now().timestamp() - cache_path.stat().st_mtime
        if cache_age < 86400 * 7:
            with open(cache_path, "r") as f:
                cached = json.load(f)
                return cached
    
    params = {
        "engine": "google_trends",
        "q": keyword,
        "date": "today 5-y",
        "api_key": api_key
    }
    
    try:
        response = requests.get(SERPAPI_BASE_URL, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        with open(cache_path, "w") as f:
            json.dump(data, f)
        
        return data
    except requests.RequestException as e:
        print(f"  [ERROR] Failed to fetch trends for '{keyword}': {e}")
        return None


def parse_trends_data(raw_data: dict) -> pd.DataFrame:
    """Parse SerpAPI Google Trends response into a DataFrame."""
    if not raw_data:
        return pd.DataFrame()
    
    timeline = raw_data.get("interest_over_time", {}).get("timeline_data", [])
    
    if not timeline:
        return pd.DataFrame()
    
    records = []
    for i, point in enumerate(timeline):
        timestamp = point.get("timestamp", "")
        if not timestamp:
            continue
        
        dt = datetime.fromtimestamp(int(timestamp))
        values = point.get("values", [])
        interest = values[0].get("extracted_value", 0) if values else 0
        
        records.append({
            "date": dt,
            "interest": interest,
            "week_num": i + 1
        })
    
    df = pd.DataFrame(records)
    
    if df.empty:
        return df
    
    df = df.sort_values("date").reset_index(drop=True)
    df["week_num"] = range(1, len(df) + 1)
    
    return df


# ============================================
# Analysis Functions (unchanged from original)
# ============================================

def calculate_trend_metrics(df: pd.DataFrame) -> dict:
    """Calculate linear regression and trend metrics."""
    if df.empty or len(df) < 10:
        return {
            "slope": None,
            "p_value": None,
            "annual_growth": None,
            "trend": "Insufficient Data"
        }
    
    x = df["week_num"].values
    y = df["interest"].values
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    annual_growth = slope * 52
    
    if p_value > P_VALUE_THRESHOLD:
        trend = "Not Significant"
    elif slope > 0:
        trend = "Growth"
    else:
        trend = "Decline"
    
    return {
        "slope": slope,
        "p_value": p_value,
        "annual_growth": annual_growth,
        "trend": trend
    }


def calculate_seasonality(df: pd.DataFrame) -> dict:
    """Calculate quarterly seasonality metrics."""
    if df.empty:
        return {"Q1": None, "Q2": None, "Q3": None, "Q4": None, "peak_quarter": None}
    
    df = df.copy()
    df["quarter"] = df["date"].dt.quarter
    
    quarterly_avg = df.groupby("quarter")["interest"].mean()
    
    q_values = {
        "Q1": quarterly_avg.get(1, 0),
        "Q2": quarterly_avg.get(2, 0),
        "Q3": quarterly_avg.get(3, 0),
        "Q4": quarterly_avg.get(4, 0)
    }
    
    peak = max(q_values, key=q_values.get)
    q_values["peak_quarter"] = peak
    
    return q_values


def calculate_monthly_seasonality(df: pd.DataFrame) -> dict:
    """Calculate monthly seasonality metrics."""
    if df.empty:
        return {m: None for m in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "peak_month"]}
    
    df = df.copy()
    df["month"] = df["date"].dt.month
    
    monthly_avg = df.groupby("month")["interest"].mean()
    
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    m_values = {}
    for i, name in enumerate(month_names, 1):
        m_values[name] = monthly_avg.get(i, 0)
    
    peak = max(month_names, key=lambda x: m_values[x])
    m_values["peak_month"] = peak
    
    return m_values


def analyze_keyword(keyword: str, api_key: str, save_to_snowflake: bool = True) -> dict:
    """Full analysis pipeline for a single keyword."""
    print(f"\nAnalyzing: {keyword}")
    
    raw_data = fetch_google_trends(keyword, api_key)
    df = parse_trends_data(raw_data)
    
    if df.empty:
        print(f"  [WARN] No data available for '{keyword}'")
        return {
            "keyword": keyword,
            "trend_metrics": {"slope": None, "p_value": None, "annual_growth": None, "trend": "No Data"},
            "seasonality": {"Q1": None, "Q2": None, "Q3": None, "Q4": None, "peak_quarter": None},
            "monthly_seasonality": {m: None for m in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                                                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "peak_month"]},
            "raw_data": pd.DataFrame()
        }
    
    # Save raw trends data to Snowflake
    if save_to_snowflake:
        try:
            save_trends_to_snowflake(keyword, df)
        except Exception as e:
            print(f"  [WARN] Could not save trends to Snowflake: {e}")
    
    trend_metrics = calculate_trend_metrics(df)
    seasonality = calculate_seasonality(df)
    monthly_seasonality = calculate_monthly_seasonality(df)
    
    p_val = trend_metrics['p_value']
    print(f"  Trend: {trend_metrics['trend']} | p-value: {p_val}")
    print(f"  Peak Quarter: {seasonality['peak_quarter']} | Peak Month: {monthly_seasonality['peak_month']}")
    
    return {
        "keyword": keyword,
        "trend_metrics": trend_metrics,
        "seasonality": seasonality,
        "monthly_seasonality": monthly_seasonality,
        "raw_data": df
    }


# ============================================
# CLI Commands
# ============================================

def cmd_upload_data():
    """Upload local CSV data to Snowflake."""
    print("\n" + "=" * 60)
    print("UMIP - Upload Data to Snowflake")
    print("=" * 60)
    
    # Upload Ahrefs data
    ahrefs_path = Path("data/ahrefs_export.csv")
    if ahrefs_path.exists():
        print(f"\nUploading Ahrefs data from {ahrefs_path}...")
        upload_ahrefs_csv_to_snowflake(str(ahrefs_path))
    else:
        print(f"[SKIP] Ahrefs file not found: {ahrefs_path}")
    
    # Upload GA4 data
    ga4_path = Path("data/ga4_export.csv")
    if ga4_path.exists():
        print(f"\nUploading GA4 data from {ga4_path}...")
        upload_ga4_csv_to_snowflake(str(ga4_path))
    else:
        print(f"[SKIP] GA4 file not found: {ga4_path}")
    
    print("\nUpload complete!")


def cmd_analyze():
    """Run trend analysis using Snowflake data."""
    print("\n" + "=" * 60)
    print("UMIP - Universal Marketing Intelligence Platform")
    print("=" * 60)
    
    config = load_config()
    
    # Load keywords from Snowflake
    print("\nLoading keywords from Snowflake...")
    keywords = load_keywords_from_snowflake()
    
    if not keywords:
        print("[ERROR] No active keywords found in Snowflake.")
        print("Run: python umip_snowflake.py upload")
        return
    
    print(f"Found {len(keywords)} keywords")
    
    # Split into cached vs uncached
    cached_keywords = [kw for kw in keywords if is_cached(kw)]
    uncached_keywords = [kw for kw in keywords if not is_cached(kw)]
    
    print(f"  Cached: {len(cached_keywords)} (instant)")
    print(f"  Need API: {len(uncached_keywords)}")
    
    results = []
    
    # Process cached keywords (instant, no API calls)
    if cached_keywords:
        print(f"\nProcessing {len(cached_keywords)} cached keywords...")
        for kw in cached_keywords:
            result = analyze_keyword(kw, config["serpapi_key"], save_to_snowflake=True)
            results.append(result)
    
    # Process uncached keywords in parallel
    if uncached_keywords:
        print(f"\nFetching {len(uncached_keywords)} keywords from API (parallel)...")
        
        def fetch_and_analyze(keyword):
            """Fetch and analyze a single keyword."""
            try:
                result = analyze_keyword(keyword, config["serpapi_key"], save_to_snowflake=True)
                return result
            except Exception as e:
                print(f"  [ERROR] {keyword}: {e}")
                return {
                    "keyword": keyword,
                    "trend_metrics": {"slope": None, "p_value": None, "annual_growth": None, "trend": "Error"},
                    "seasonality": {"Q1": None, "Q2": None, "Q3": None, "Q4": None, "peak_quarter": None},
                    "monthly_seasonality": {m: None for m in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                                                               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "peak_month"]},
                    "raw_data": pd.DataFrame()
                }
        
        # Use 5 parallel workers
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(fetch_and_analyze, kw): kw for kw in uncached_keywords}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
    
    # Save analysis results to Snowflake
    print("\nSaving analysis results to Snowflake...")
    save_analysis_to_snowflake(results)
    
    # Summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    
    growth_count = sum(1 for r in results if r["trend_metrics"]["trend"] == "Growth")
    decline_count = sum(1 for r in results if r["trend_metrics"]["trend"] == "Decline")
    ns_count = sum(1 for r in results if r["trend_metrics"]["trend"] == "Not Significant")
    no_data_count = sum(1 for r in results if r["trend_metrics"]["trend"] in ["No Data", "Error"])
    
    print(f"  Growth trends:         {growth_count}")
    print(f"  Decline trends:        {decline_count}")
    print(f"  Not Significant:       {ns_count}")
    if no_data_count > 0:
        print(f"  No Data/Error:         {no_data_count}")
    print(f"\nResults saved to Snowflake PRIORITY_TIRE_DATA.UMIP schema")


def cmd_query_results():
    """Query and display recent analysis results from Snowflake."""
    print("\n" + "=" * 60)
    print("UMIP - Recent Analysis Results")
    print("=" * 60)
    
    query = """
        SELECT 
            t.KEYWORD,
            t.TREND_CLASSIFICATION,
            t.SLOPE,
            t.P_VALUE,
            t.ANNUAL_GROWTH,
            q.PEAK_QUARTER,
            m.PEAK_MONTH,
            t.ANALYSIS_DATE
        FROM PRIORITY_TIRE_DATA.UMIP.TREND_ANALYSIS t
        LEFT JOIN PRIORITY_TIRE_DATA.UMIP.SEASONALITY_QUARTERLY q 
            ON t.KEYWORD = q.KEYWORD AND t.ANALYSIS_DATE = q.ANALYSIS_DATE
        LEFT JOIN PRIORITY_TIRE_DATA.UMIP.SEASONALITY_MONTHLY m
            ON t.KEYWORD = m.KEYWORD AND t.ANALYSIS_DATE = m.ANALYSIS_DATE
        WHERE t.ANALYSIS_DATE = (SELECT MAX(ANALYSIS_DATE) FROM PRIORITY_TIRE_DATA.UMIP.TREND_ANALYSIS)
        ORDER BY t.KEYWORD
    """
    
    df = execute_query(query)
    
    if df.empty:
        print("\nNo analysis results found. Run: python umip_snowflake.py analyze")
        return
    
    print(f"\nResults from {df['ANALYSIS_DATE'].iloc[0]}:\n")
    print(df.to_string(index=False))


def cmd_status():
    """Show Snowflake data status."""
    print("\n" + "=" * 60)
    print("UMIP - Snowflake Data Status")
    print("=" * 60)
    
    tables = [
        ("AHREFS_KEYWORDS", "Ahrefs keyword records"),
        ("GA4_PAGE_METRICS", "GA4 page records"),
        ("GOOGLE_TRENDS_TIMESERIES", "Trends data points"),
        ("TREND_ANALYSIS", "Trend analysis records"),
        ("SEASONALITY_QUARTERLY", "Quarterly seasonality records"),
        ("SEASONALITY_MONTHLY", "Monthly seasonality records"),
    ]
    
    print("\nTable row counts:")
    for table, desc in tables:
        try:
            df = execute_query(f"SELECT COUNT(*) as CNT FROM PRIORITY_TIRE_DATA.UMIP.{table}")
            count = df["CNT"].iloc[0] if not df.empty else 0
            print(f"  {desc}: {count}")
        except Exception as e:
            print(f"  {desc}: [ERROR] {e}")


def main():
    """Main entry point with CLI commands."""
    import sys
    
    commands = {
        "upload": cmd_upload_data,
        "analyze": cmd_analyze,
        "results": cmd_query_results,
        "status": cmd_status,
    }
    
    if len(sys.argv) < 2:
        print("UMIP - Universal Marketing Intelligence Platform")
        print("\nUsage: python umip_snowflake.py <command>")
        print("\nCommands:")
        print("  upload   - Upload local CSV data to Snowflake")
        print("  analyze  - Run trend analysis (reads from Snowflake)")
        print("  results  - Query recent analysis results")
        print("  status   - Show Snowflake data status")
        return
    
    cmd = sys.argv[1].lower()
    if cmd in commands:
        commands[cmd]()
    else:
        print(f"Unknown command: {cmd}")
        print(f"Available commands: {', '.join(commands.keys())}")


if __name__ == "__main__":
    main()
