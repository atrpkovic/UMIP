"""
UMIP Frontend - Shared Utilities
"""
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
# Try Snowflake Streamlit context first, fall back to connector
try:
    from snowflake.snowpark.context import get_active_session
    IN_SNOWFLAKE = True
except ImportError:
    IN_SNOWFLAKE = False
    import snowflake.connector


def get_snowflake_connection():
    """Get Snowflake connection - works in Streamlit-in-Snowflake or locally."""
    if IN_SNOWFLAKE:
        return get_active_session()
    else:
        return snowflake.connector.connect(
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "GSHOPPING_WH"),
            database=os.getenv("SNOWFLAKE_DATABASE", "PRIORITY_TIRE_DATA"),
            schema=os.getenv("SNOWFLAKE_SCHEMA", "UMIP")
        )


@st.cache_data(ttl=3600)
def load_keyword_analysis():
    """Load the joined keyword analysis view."""
    query = """
    SELECT * FROM PRIORITY_TIRE_DATA.UMIP.KEYWORD_ANALYSIS
    """
    
    if IN_SNOWFLAKE:
        session = get_snowflake_connection()
        df = session.sql(query).to_pandas()
    else:
        conn = get_snowflake_connection()
        df = pd.read_sql(query, conn)
        conn.close()
    
    return df


@st.cache_data(ttl=3600)
def load_trend_timeseries(keyword: str):
    """Load 5-year trend data for a specific keyword."""
    query = f"""
    SELECT TREND_DATE, INTEREST 
    FROM PRIORITY_TIRE_DATA.UMIP.GOOGLE_TRENDS_TIMESERIES
    WHERE KEYWORD = '{keyword}'
    ORDER BY TREND_DATE
    """
    
    if IN_SNOWFLAKE:
        session = get_snowflake_connection()
        df = session.sql(query).to_pandas()
    else:
        conn = get_snowflake_connection()
        df = pd.read_sql(query, conn)
        conn.close()
    
    return df


def calculate_opportunity_score(row, weights):
    """
    Calculate opportunity score based on:
    - Trend momentum (higher = better)
    - Search volume (higher = better)  
    - Keyword difficulty (lower = better)
    
    All factors normalized to 0-1 scale.
    """
    trend_score = 0
    volume_score = 0
    competition_score = 0
    
    # Trend momentum - use annual_growth, normalize assuming -50 to +50 range
    if pd.notna(row.get("ANNUAL_GROWTH")):
        annual_growth = row["ANNUAL_GROWTH"]
        trend_score = (annual_growth + 50) / 100  # Maps -50 to 0, +50 to 1
        trend_score = max(0, min(1, trend_score))  # Clamp to 0-1
    
    # Volume - log scale normalization (assuming 100 to 100k range)
    if pd.notna(row.get("SEARCH_VOLUME")) and row["SEARCH_VOLUME"] > 0:
        import math
        vol = row["SEARCH_VOLUME"]
        volume_score = (math.log10(vol) - 2) / 3  # log10(100)=2, log10(100k)=5
        volume_score = max(0, min(1, volume_score))
    
    # Competition - invert KD (0-100 scale, lower is better)
    if pd.notna(row.get("KD")):
        competition_score = 1 - (row["KD"] / 100)
    
    # Weighted average
    score = (
        weights["trend"] * trend_score +
        weights["volume"] * volume_score +
        weights["competition"] * competition_score
    )
    
    return round(score * 100, 1)  # Return as 0-100 score


def get_trend_emoji(trend_class):
    """Return emoji for trend classification."""
    if trend_class == "Growth":
        return "+"
    elif trend_class == "Decline":
        return "-"
    else:
        return "~"


def format_number(val):
    """Format large numbers with K/M suffix."""
    if pd.isna(val):
        return "-"
    if val >= 1_000_000:
        return f"{val/1_000_000:.1f}M"
    elif val >= 1_000:
        return f"{val/1_000:.1f}K"
    else:
        return str(int(val))
    
def calculate_seasonality_score(row, current_month=None):
    """
    Calculate Seasonal Priority score based on:
    - Amplitude (30%): How much interest swings between peak and trough
    - Time to peak (40%): Urgency - peaks soon = higher score
    - Peak consistency (20%): How predictable is the pattern
    - Trend direction (10%): Bonus for growth, penalty for decline
    
    Returns score 0-100.
    """
    import datetime
    
    if current_month is None:
        current_month = datetime.datetime.now().month
    
    amplitude_score = 0
    time_score = 0
    consistency_score = 0
    trend_score = 0.5  # neutral default
    
    # --- Amplitude (30%) ---
    # Get monthly averages to find peak and trough
    month_cols = ['JAN_AVG', 'FEB_AVG', 'MAR_AVG', 'APR_AVG', 'MAY_AVG', 'JUN_AVG',
                  'JUL_AVG', 'AUG_AVG', 'SEP_AVG', 'OCT_AVG', 'NOV_AVG', 'DEC_AVG']
    
    monthly_vals = []
    for col in month_cols:
        val = row.get(col)
        if pd.notna(val) and val > 0:
            monthly_vals.append(val)
    
    if len(monthly_vals) >= 2:
        peak_val = max(monthly_vals)
        trough_val = min(monthly_vals)
        if trough_val > 0:
            amplitude = (peak_val - trough_val) / trough_val
            # Normalize: 0x = 0, 5x+ = 1
            amplitude_score = min(amplitude / 5, 1)
    
    # --- Time to peak (40%) ---
    peak_month_str = row.get("PEAK_MONTH")
    if pd.notna(peak_month_str):
        month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                     'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
        peak_month = month_map.get(peak_month_str, 0)
        
        if peak_month > 0:
            # Months until peak (circular)
            months_until = (peak_month - current_month) % 12
            if months_until == 0:
                months_until = 12  # Peak is this month, treat as 12 months away for next cycle
            
            # Closer = higher score: 1 month away = 1.0, 12 months away = 0.08
            time_score = (12 - months_until + 1) / 12
    
    # --- Peak consistency (20%) ---
    consistency = row.get("PEAK_CONSISTENCY")
    if pd.notna(consistency):
        consistency_score = consistency  # Already 0-1
    
    # --- Trend direction (10%) ---
    trend = row.get("TREND_CLASSIFICATION")
    if trend == "Growth":
        trend_score = 1.0
    elif trend == "Decline":
        trend_score = 0.0
    else:
        trend_score = 0.5
    
    # Weighted combination
    score = (
        0.30 * amplitude_score +
        0.40 * time_score +
        0.20 * consistency_score +
        0.10 * trend_score
    ) * 100
    
    return round(score, 1)

def get_months_until_peak(peak_month_str, current_month=None):
    """Returns months until peak and urgency label."""
    import datetime
    
    if current_month is None:
        current_month = datetime.datetime.now().month
    
    if pd.isna(peak_month_str):
        return None, None
    
    month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    peak_month = month_map.get(peak_month_str, 0)
    
    if peak_month == 0:
        return None, None
    
    months_until = (peak_month - current_month) % 12
    if months_until == 0:
        months_until = 12
    
    if months_until <= 2:
        urgency = "Act Now"
    elif months_until <= 4:
        urgency = "Coming Soon"
    else:
        urgency = "Plan Ahead"
    
    return months_until, urgency