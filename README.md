# UMIP - Universal Marketing Intelligence Platform

Centralized marketing intelligence combining Google Trends analysis with Ahrefs and GA4 data, stored in Snowflake.

## Data Flow

```
Your Exports                     Snowflake                              Output
-----------                      ---------                              ------
data/ahrefs_export.csv    →    AHREFS_KEYWORDS     ─┐
data/ga4_export.csv       →    GA4_PAGE_METRICS     ├─→  KEYWORD_ANALYSIS (VIEW)
data/keywords_master.csv  →    KEYWORDS_MASTER      │
         ↓                                          │
    Google Trends API     →    TREND_ANALYSIS      ─┤
                               SEASONALITY_*       ─┘
```

## File Structure

```
UMIP/
├── data/
│   ├── ahrefs_export.csv      ← Put your Ahrefs export here
│   ├── ga4_export.csv         ← Put your GA4 export here
│   └── keywords_master.csv    ← Keywords for trend analysis
├── umip_snowflake.py          ← Main script
├── .env                       ← Your credentials (don't commit!)
└── setup_snowflake.sql        ← Already ran this
```

## Expected File Formats

### ahrefs_export.csv
Export from Ahrefs Site Explorer → Organic Keywords. Must include columns:
- Keyword
- Current URL
- Volume
- Current position
- Current traffic
- Keyword Difficulty
- Cost per click

### ga4_export.csv
Export from GA4 Explorations. Must include columns:
- Landing page
- Sessions
- User key event rate
- Total revenue

### keywords_master.csv
Simple CSV with keywords to analyze trends for:
```
keyword
winter tires
fullway tires
tractor tires
```

## Usage

### Upload Data
```bash
python umip_snowflake.py upload
```

### Run Trend Analysis
```bash
python umip_snowflake.py analyze
```

### View Results
```bash
python umip_snowflake.py results
```

### Check Status
```bash
python umip_snowflake.py status
```

## Query the Joined Data in Snowflake

```sql
SELECT * FROM PRIORITY_TIRE_DATA.UMIP.KEYWORD_ANALYSIS
ORDER BY SEARCH_VOLUME DESC;
```

## Snowflake Tables

| Table | Description |
|-------|-------------|
| AHREFS_KEYWORDS | Keyword data with URL mapping |
| GA4_PAGE_METRICS | Page performance metrics |
| KEYWORDS_MASTER | Keywords to run trend analysis on |
| GOOGLE_TRENDS_TIMESERIES | Raw 5-year trends data |
| TREND_ANALYSIS | Slope, p-value, growth classification |
| SEASONALITY_QUARTERLY | Q1-Q4 averages |
| SEASONALITY_MONTHLY | Jan-Dec averages |
| KEYWORD_ANALYSIS | **VIEW** - Joins everything together |
