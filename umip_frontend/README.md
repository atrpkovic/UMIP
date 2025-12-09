# UMIP Frontend

Streamlit dashboard for the Universal Marketing Intelligence Platform. Helps PPC team find rising keyword trends to bid on.

## Features

- **Opportunity Finder** - Ranked keyword list with configurable scoring
- **Keyword Deep Dive** - 5-year trend chart, seasonality breakdown, GA4 metrics
- **CSV Export** - Download filtered results for Google Ads import

## Setup

### Local Development

1. Copy `.env.example` to `.env` and fill in your Snowflake credentials
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run:
   ```bash
   streamlit run Home.py
   ```

### Streamlit-in-Snowflake

1. In Snowsight, go to Streamlit > Create Streamlit App
2. Upload `Home.py`, `utils.py`, and the `pages/` folder
3. The app auto-detects the Snowflake session - no credentials needed

## Data Requirements

The app expects these tables/views in `PRIORITY_TIRE_DATA.UMIP`:

- `KEYWORD_ANALYSIS` (VIEW) - joined data with all metrics
- `GOOGLE_TRENDS_TIMESERIES` - 5-year trend data per keyword

### Expected columns in KEYWORD_ANALYSIS:

| Column | Source | Description |
|--------|--------|-------------|
| KEYWORD | Ahrefs | Keyword text |
| SEARCH_VOLUME | Ahrefs | Monthly search volume |
| KD | Ahrefs | Keyword difficulty (0-100) |
| TREND_CLASSIFICATION | Trends | Growth / Decline / Not Significant |
| ANNUAL_GROWTH | Trends | Estimated annual change in interest |
| SLOPE | Trends | Linear regression slope |
| P_VALUE | Trends | Statistical significance |
| Q1_AVG, Q2_AVG, Q3_AVG, Q4_AVG | Trends | Quarterly averages |
| PEAK_QUARTER | Trends | Highest traffic quarter |
| SESSIONS | GA4 | Page sessions |
| REVENUE | GA4 | Page revenue |
| KEY_EVENT_RATE | GA4 | Conversion rate |

## Opportunity Score

The score combines three factors (weights adjustable in sidebar):

1. **Trend momentum** (default 50%) - Higher annual growth = higher score
2. **Search volume** (default 30%) - Log-scaled, bigger volume = higher score  
3. **Low competition** (default 20%) - Lower keyword difficulty = higher score

Score ranges from 0-100. Higher = better opportunity.

## Usage Tips

1. Start with default filters to see top opportunities
2. Adjust "Min annual growth" slider to focus on growing keywords
3. Click a row to select, then visit Keyword Detail for the full picture
4. Export filtered list to CSV for Google Ads upload
