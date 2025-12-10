"""
UMIP - Universal Marketing Intelligence Platform
Seasonal Opportunity Finder for PPC Team
"""
import streamlit as st
import pandas as pd
from utils import (
    load_keyword_analysis, 
    calculate_seasonality_score,
    get_months_until_peak,
    get_trend_emoji,
    format_number
)

# Page config
st.set_page_config(
    page_title="UMIP - Seasonal Priorities",
    page_icon="chart_with_upwards_trend",
    layout="wide"
)

st.title("UMIP Seasonal Priority Finder")
st.caption("Find keywords peaking soon to bid on before competitors catch on")

# Load data
with st.spinner("Loading keyword data..."):
    df = load_keyword_analysis()

if df.empty:
    st.error("No data found. Make sure KEYWORD_ANALYSIS view is populated.")
    st.stop()

# Calculate seasonality scores and urgency for all rows
df["SEASONAL_PRIORITY"] = df.apply(calculate_seasonality_score, axis=1)
df["MONTHS_UNTIL_PEAK"], df["URGENCY"] = zip(*df["PEAK_MONTH"].apply(get_months_until_peak))

# Sidebar filters
st.sidebar.header("Filters")

# Keyword search
keyword_list = [""] + sorted(df["KEYWORD"].dropna().unique().tolist())
keyword_search = st.sidebar.selectbox(
    "Search keywords",
    keyword_list,
    index=0,
    help="Select a keyword or type to filter"
)

# Urgency filter
urgency_options = ["All", "Act Now", "Coming Soon", "Plan Ahead"]
urgency_filter = st.sidebar.selectbox(
    "Urgency",
    urgency_options,
    help="Act Now = peaks within 2 months, Coming Soon = 3-4 months, Plan Ahead = 5+ months"
)

# Trend filter
trend_options = ["All", "Growth", "Decline", "Not Significant", "No Data"]
trend_filter = st.sidebar.selectbox(
    "Trend direction", 
    trend_options,
    help="Growth = rising interest, Decline = falling interest, Not Significant = no clear trend"
)

# Minimum volume
min_volume = st.sidebar.number_input(
    "Min search volume", 
    min_value=0, 
    max_value=100000, 
    value=0,
    step=100,
    help="Only show keywords with at least this many monthly searches"
)

st.sidebar.divider()
st.sidebar.caption("**Score weights:** Amplitude 30%, Time to Peak 40%, Consistency 20%, Trend 10%")

# Apply filters
filtered_df = df.copy()

if keyword_search:
    filtered_df = filtered_df[
        filtered_df["KEYWORD"].str.contains(keyword_search, case=False, na=False)
    ]

if urgency_filter != "All":
    filtered_df = filtered_df[filtered_df["URGENCY"] == urgency_filter]

if trend_filter != "All":
    if trend_filter == "No Data":
        filtered_df = filtered_df[filtered_df["TREND_CLASSIFICATION"].isna() | (filtered_df["TREND_CLASSIFICATION"] == "No Data")]
    else:
        filtered_df = filtered_df[filtered_df["TREND_CLASSIFICATION"] == trend_filter]

if "SEARCH_VOLUME" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["SEARCH_VOLUME"] >= min_volume]

# Sort by seasonal priority
filtered_df = filtered_df.sort_values("SEASONAL_PRIORITY", ascending=False)

# Display metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "Keywords Found", 
        len(filtered_df),
        help="Total keywords matching your filters"
    )
with col2:
    act_now_count = len(filtered_df[filtered_df["URGENCY"] == "Act Now"])
    st.metric(
        "Act Now", 
        act_now_count,
        help="Keywords peaking within 2 months"
    )
with col3:
    if len(filtered_df) > 0:
        avg_score = filtered_df["SEASONAL_PRIORITY"].mean()
        st.metric(
            "Avg Priority Score", 
            f"{avg_score:.1f}",
            help="Average seasonal priority score (0-100)"
        )
    else:
        st.metric("Avg Priority Score", "-")
with col4:
    if len(filtered_df) > 0 and "SEARCH_VOLUME" in filtered_df.columns:
        total_vol = filtered_df["SEARCH_VOLUME"].sum()
        st.metric(
            "Total Search Volume", 
            format_number(total_vol),
            help="Combined monthly searches for all filtered keywords"
        )
    else:
        st.metric("Total Search Volume", "-")

st.divider()

# Main table
if len(filtered_df) == 0:
    st.warning("No keywords match your filters.")
else:
    # Prepare display columns
    display_cols = ["KEYWORD", "SEASONAL_PRIORITY", "URGENCY", "PEAK_MONTH", "MONTHS_UNTIL_PEAK",
                    "TREND_CLASSIFICATION", "SEARCH_VOLUME", "PEAK_CONSISTENCY"]
    
    # Filter to available columns
    display_cols = [c for c in display_cols if c in filtered_df.columns]
    
    display_df = filtered_df[display_cols].copy()
    
    # Rename for display
    column_renames = {
        "KEYWORD": "Keyword",
        "SEASONAL_PRIORITY": "Priority",
        "URGENCY": "Urgency",
        "PEAK_MONTH": "Peak Month",
        "MONTHS_UNTIL_PEAK": "Months Away",
        "TREND_CLASSIFICATION": "Trend",
        "SEARCH_VOLUME": "Volume",
        "PEAK_CONSISTENCY": "Consistency"
    }
    display_df = display_df.rename(columns=column_renames)
    
    # Format consistency as percentage
    if "Consistency" in display_df.columns:
        display_df["Consistency"] = display_df["Consistency"].apply(
            lambda x: f"{x:.0%}" if pd.notna(x) else "-"
        )
    
    # Add trend indicator
    if "Trend" in display_df.columns:
        display_df["Trend"] = display_df["Trend"].apply(
            lambda x: f"{get_trend_emoji(x)} {x}" if pd.notna(x) else "~ No Data"
        )
    
    st.subheader(f"Seasonal Priorities ({len(display_df)} keywords)")
    
    # Use dataframe with selection
    # Use dataframe with selection
    event = st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Priority": st.column_config.ProgressColumn(
                "Priority",
                help="Seasonal priority score based on amplitude, timing, consistency, and trend (0-100)",
                min_value=0,
                max_value=100,
                format="%.0f"
            ),
            "Keyword": st.column_config.TextColumn(
                "Keyword",
                help="The search term"
            ),
            "Urgency": st.column_config.TextColumn(
                "Urgency",
                help="Act Now = peaks within 2 months, Coming Soon = 3-4 months, Plan Ahead = 5+ months"
            ),
            "Peak Month": st.column_config.TextColumn(
                "Peak Month",
                help="Month with highest average search interest"
            ),
            "Months Away": st.column_config.NumberColumn(
                "Months Away",
                help="Months until peak season",
                format="%d"
            ),
            "Trend": st.column_config.TextColumn(
                "Trend",
                help="Long-term trend direction over 5 years"
            ),
            "Volume": st.column_config.NumberColumn(
                "Volume",
                help="Monthly search volume from Ahrefs",
                format="%d"
            ),
            "Consistency": st.column_config.TextColumn(
                "Consistency",
                help="How often this keyword peaks in the same month year-over-year"
            )
        }
    )