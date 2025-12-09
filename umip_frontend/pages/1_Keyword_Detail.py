"""
UMIP - Keyword Detail Page
Deep dive on a single keyword: trend chart, seasonality, metrics
"""
import streamlit as st
import pandas as pd
from utils import (
    load_keyword_analysis,
    load_trend_timeseries,
    get_trend_emoji,
    format_number
)

st.set_page_config(
    page_title="UMIP - Keyword Detail",
    page_icon="mag",
    layout="wide"
)

st.title("Keyword Deep Dive")

# Load keyword list for selector
df = load_keyword_analysis()

if df.empty:
    st.error("No data found.")
    st.stop()

keywords = sorted(df["KEYWORD"].dropna().unique().tolist())

# Get selected keyword from session state or default to first
default_keyword = st.session_state.get("selected_keyword", keywords[0] if keywords else None)
default_idx = keywords.index(default_keyword) if default_keyword in keywords else 0

col1, col2 = st.columns([1, 2])
with col1:
    selected_keyword = st.selectbox(
        "Select keyword",
        keywords,
        index=default_idx
    )

if not selected_keyword:
    st.warning("No keyword selected.")
    st.stop()

# Get keyword data
kw_data = df[df["KEYWORD"] == selected_keyword].iloc[0]

# Header with trend indicator
trend_class = kw_data.get("TREND_CLASSIFICATION", "Unknown")
trend_emoji = get_trend_emoji(trend_class)

st.header(f"{trend_emoji} {selected_keyword}")

# Key metrics row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    vol = kw_data.get("SEARCH_VOLUME")
    st.metric("Search Volume", format_number(vol) if pd.notna(vol) else "-")

with col2:
    growth = kw_data.get("ANNUAL_GROWTH")
    if pd.notna(growth):
        delta_color = "normal" if growth >= 0 else "inverse"
        st.metric("Annual Growth", f"{growth:+.1f}", delta=f"{growth:+.1f} pts/yr", delta_color=delta_color)
    else:
        st.metric("Annual Growth", "-")

with col3:
    st.metric("Trend", trend_class)

with col4:
    kd = kw_data.get("KD")
    st.metric("Keyword Difficulty", f"{int(kd)}" if pd.notna(kd) else "-")

with col5:
    peak = kw_data.get("PEAK_QUARTER")
    st.metric("Peak Quarter", peak if pd.notna(peak) else "-")

st.divider()

# Trend chart
st.subheader("5-Year Trend")

trend_data = load_trend_timeseries(selected_keyword)

if trend_data.empty:
    st.warning("No trend timeseries data available for this keyword.")
else:
    trend_data["TREND_DATE"] = pd.to_datetime(trend_data["TREND_DATE"])
    trend_data = trend_data.sort_values("TREND_DATE")
    
    st.line_chart(
        trend_data,
        x="TREND_DATE",
        y="INTEREST",
        use_container_width=True
    )
    
    # Trend stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        slope = kw_data.get("SLOPE")
        if pd.notna(slope):
            st.metric("Slope", f"{slope:.4f}", help="Change in interest per week")
    
    with col2:
        p_val = kw_data.get("P_VALUE")
        if pd.notna(p_val):
            significance = "Significant" if p_val < 0.05 else "Not Significant"
            if p_val < 0.0001:
                p_display = "< 0.0001"
            else:
                p_display = f"{p_val:.4f}"
            st.metric("p-value", p_display, help=f"Statistical significance: {significance}")
    
    with col3:
        # Current vs 1 year ago
        if len(trend_data) >= 52:
            current = trend_data["INTEREST"].iloc[-1]
            year_ago = trend_data["INTEREST"].iloc[-52]
            yoy_change = current - year_ago
            st.metric("YoY Change", f"{yoy_change:+.0f} pts", delta=f"{yoy_change:+.0f}")

st.divider()

# Seasonality breakdown
st.subheader("Seasonality")

q1 = kw_data.get("Q1_AVG") or kw_data.get("Q1")
q2 = kw_data.get("Q2_AVG") or kw_data.get("Q2")
q3 = kw_data.get("Q3_AVG") or kw_data.get("Q3")
q4 = kw_data.get("Q4_AVG") or kw_data.get("Q4")

if all(pd.notna(x) for x in [q1, q2, q3, q4]):
    seasonality_df = pd.DataFrame({
        "Quarter": ["Q1 (Jan-Mar)", "Q2 (Apr-Jun)", "Q3 (Jul-Sep)", "Q4 (Oct-Dec)"],
        "Avg Interest": [q1, q2, q3, q4]
    })
    
    st.bar_chart(
        seasonality_df,
        x="Quarter",
        y="Avg Interest",
        use_container_width=True
    )
    
    # Seasonality insight
    peak_quarter = kw_data.get("PEAK_QUARTER")
    quarters = {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4}
    
    if peak_quarter:
        peak_val = quarters.get(peak_quarter, 0)
        min_quarter = min(quarters, key=quarters.get)
        min_val = quarters[min_quarter]
        
        if min_val > 0:
            ratio = peak_val / min_val
            st.info(f"**Seasonality insight:** {peak_quarter} sees **{ratio:.1f}x** more interest than {min_quarter}. Plan campaigns 4-6 weeks before {peak_quarter} starts.")
else:
    st.info("Seasonality data not available for this keyword.")

st.divider()

# GA4 performance (if available)
st.subheader("Page Performance (GA4)")

sessions = kw_data.get("SESSIONS")
revenue = kw_data.get("REVENUE")
key_event_rate = kw_data.get("KEY_EVENT_RATE") or kw_data.get("USER_KEY_EVENT_RATE")
page_path = kw_data.get("PAGE_PATH") or kw_data.get("CURRENT_URL")

if pd.notna(sessions) or pd.notna(revenue):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Sessions", format_number(sessions) if pd.notna(sessions) else "-")
    
    with col2:
        if pd.notna(revenue):
            st.metric("Revenue", f"${revenue:,.0f}")
        else:
            st.metric("Revenue", "-")
    
    with col3:
        if pd.notna(key_event_rate):
            st.metric("Key Event Rate", f"{key_event_rate:.1%}" if key_event_rate < 1 else f"{key_event_rate:.1f}%")
        else:
            st.metric("Key Event Rate", "-")
    
    if pd.notna(page_path):
        st.caption(f"Landing page: `{page_path}`")
else:
    st.info("No GA4 data linked to this keyword. Check the keyword-to-page mapping.")

st.divider()

# Raw data expander
with st.expander("View raw data"):
    st.json(kw_data.to_dict())
