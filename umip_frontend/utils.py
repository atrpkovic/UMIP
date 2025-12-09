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
