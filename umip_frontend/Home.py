"""
UMIP - Universal Marketing Intelligence Platform
Opportunity Finder for PPC Team
"""
import streamlit as st
import pandas as pd
from utils import (
    load_keyword_analysis, 
    calculate_opportunity_score,
    get_trend_emoji,
    format_number
)

# Page config
st.set_page_config(
    page_title="UMIP - Opportunities",
    page_icon="chart_with_upwards_trend",
    layout="wide"
)

st.title("UMIP Opportunity Finder")
st.caption("Find rising keywords to bid on before competitors catch on")

# Load data
with st.spinner("Loading keyword data..."):
    df = load_keyword_analysis()

if df.empty:
    st.error("No data found. Make sure KEYWORD_ANALYSIS view is populated.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")

# Keyword search
keyword_search = st.sidebar.text_input(
    "Search keywords", 
    placeholder="e.g. winter tires",
    help="Filter keywords containing this text"
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

# Score weights in expander to reduce clutter
with st.sidebar.expander("Score Weights", expanded=False):
    st.caption("Adjust how factors contribute to opportunity score")
    
    w_trend = st.slider(
        "Trend momentum", 0.0, 1.0, 0.5, 0.1,
        help="How much rising trends affect the score"
    )
    w_volume = st.slider(
        "Search volume", 0.0, 1.0, 0.3, 0.1,
        help="How much search volume affects the score"
    )
    w_competition = st.slider(
        "Low competition", 0.0, 1.0, 0.2, 0.1,
        help="How much low keyword difficulty affects the score"
    )

# Normalize weights
total_weight = w_trend + w_volume + w_competition
if total_weight > 0:
    weights = {
        "trend": w_trend / total_weight,
        "volume": w_volume / total_weight,
        "competition": w_competition / total_weight
    }
else:
    weights = {"trend": 0.33, "volume": 0.33, "competition": 0.34}

# Apply filters
filtered_df = df.copy()

if keyword_search:
    filtered_df = filtered_df[
        filtered_df["KEYWORD"].str.contains(keyword_search, case=False, na=False)
    ]

if trend_filter != "All":
    if trend_filter == "No Data":
        filtered_df = filtered_df[filtered_df["TREND_CLASSIFICATION"].isna()]
    else:
        filtered_df = filtered_df[filtered_df["TREND_CLASSIFICATION"] == trend_filter]

if "SEARCH_VOLUME" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["SEARCH_VOLUME"] >= min_volume]

# Calculate opportunity scores
filtered_df["OPPORTUNITY_SCORE"] = filtered_df.apply(
    lambda row: calculate_opportunity_score(row, weights), axis=1
)

# Sort by opportunity score
filtered_df = filtered_df.sort_values("OPPORTUNITY_SCORE", ascending=False)

# Display metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "Keywords Found", 
        len(filtered_df),
        help="Total keywords matching your filters"
    )
with col2:
    growth_count = len(filtered_df[filtered_df["TREND_CLASSIFICATION"] == "Growth"])
    st.metric(
        "Growing Trends", 
        growth_count,
        help="Keywords with statistically significant upward trends"
    )
with col3:
    if len(filtered_df) > 0:
        avg_score = filtered_df["OPPORTUNITY_SCORE"].mean()
        st.metric(
            "Avg Opportunity Score", 
            f"{avg_score:.1f}",
            help="Average score across filtered keywords (0-100)"
        )
    else:
        st.metric("Avg Opportunity Score", "-")
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
    display_cols = ["KEYWORD", "OPPORTUNITY_SCORE", "TREND_CLASSIFICATION", "ANNUAL_GROWTH", 
                    "SEARCH_VOLUME", "KD", "PEAK_QUARTER"]
    
    # Filter to available columns
    display_cols = [c for c in display_cols if c in filtered_df.columns]
    
    display_df = filtered_df[display_cols].copy()
    
    # Rename for display
    column_renames = {
        "KEYWORD": "Keyword",
        "OPPORTUNITY_SCORE": "Score",
        "TREND_CLASSIFICATION": "Trend",
        "ANNUAL_GROWTH": "Annual Growth",
        "SEARCH_VOLUME": "Volume",
        "KD": "Difficulty",
        "PEAK_QUARTER": "Peak Quarter"
    }
    display_df = display_df.rename(columns=column_renames)
    
    # Format annual growth with sign
    if "Annual Growth" in display_df.columns:
        display_df["Annual Growth"] = display_df["Annual Growth"].apply(
            lambda x: f"+{x:.1f}" if pd.notna(x) and x > 0 else (f"{x:.1f}" if pd.notna(x) else "-")
        )
    
    # Add trend indicator
    if "Trend" in display_df.columns:
        display_df["Trend"] = display_df["Trend"].apply(
            lambda x: f"{get_trend_emoji(x)} {x}" if pd.notna(x) else "~ No Data"
        )
    
    # Store selected keyword in session state for detail page
    st.subheader(f"Top Opportunities ({len(display_df)} keywords)")
    
    # Use dataframe with selection
    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score",
                help="Opportunity score based on trend, volume, and competition (0-100)",
                min_value=0,
                max_value=100,
                format="%.0f"
            ),
            "Keyword": st.column_config.TextColumn(
                "Keyword",
                help="The search term"
            ),
            "Trend": st.column_config.TextColumn(
                "Trend",
                help="Growth = rising searches, Decline = falling, ~ = no clear trend"
            ),
            "Annual Growth": st.column_config.TextColumn(
                "Annual Growth",
                help="Estimated yearly change in Google Trends interest points"
            ),
            "Volume": st.column_config.NumberColumn(
                "Volume",
                help="Monthly search volume from Ahrefs",
                format="%d"
            ),
            "Difficulty": st.column_config.NumberColumn(
                "Difficulty",
                help="Keyword Difficulty (0-100) - lower is easier to rank",
                format="%d"
            ),
            "Peak Quarter": st.column_config.TextColumn(
                "Peak Quarter",
                help="Quarter with highest search interest (Q1=Jan-Mar, Q2=Apr-Jun, etc.)"
            )
        }
    )
    
    # Handle row selection
    if event.selection and event.selection.rows:
        selected_idx = event.selection.rows[0]
        selected_keyword = filtered_df.iloc[selected_idx]["KEYWORD"]
        st.session_state["selected_keyword"] = selected_keyword
        st.info(f"Selected: **{selected_keyword}** - Go to Keyword Detail page for deep dive")
    
    st.divider()
    
    # Export section
    st.subheader("Export")
    
    export_df = filtered_df.copy()
    if "OPPORTUNITY_SCORE" in export_df.columns:
        # Reorder with score first
        cols = ["KEYWORD", "OPPORTUNITY_SCORE"] + [c for c in export_df.columns if c not in ["KEYWORD", "OPPORTUNITY_SCORE"]]
        export_df = export_df[cols]
    
    csv = export_df.to_csv(index=False)
    st.download_button(
        label="Download CSV for Google Ads",
        data=csv,
        file_name="umip_opportunities.csv",
        mime="text/csv",
        help="Export filtered keywords with all metrics"
    )