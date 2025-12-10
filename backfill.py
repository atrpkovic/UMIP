"""
Backfill PEAK_CONSISTENCY for existing keywords in SEASONALITY_MONTHLY
"""
import os
import pandas as pd
from dotenv import load_dotenv
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

load_dotenv()

def get_connection():
    return snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "GSHOPPING_WH"),
        database="PRIORITY_TIRE_DATA",
        schema="UMIP"
    )

def calculate_peak_consistency(keyword: str, conn) -> float:
    """
    Calculate what % of years the keyword peaked in the same month.
    Returns value between 0 and 1.
    """
    query = f"""
    SELECT 
        YEAR(TREND_DATE) as year,
        MONTH(TREND_DATE) as month,
        AVG(INTEREST) as avg_interest
    FROM GOOGLE_TRENDS_TIMESERIES
    WHERE KEYWORD = '{keyword}'
    GROUP BY YEAR(TREND_DATE), MONTH(TREND_DATE)
    ORDER BY year, month
    """
    
    df = pd.read_sql(query, conn)
    
    if df.empty:
        return None
    
    # Find peak month for each year
    peak_months_by_year = df.loc[df.groupby('YEAR')['AVG_INTEREST'].idxmax()]['MONTH'].tolist()
    
    if not peak_months_by_year:
        return None
    
    # Find most common peak month
    from collections import Counter
    month_counts = Counter(peak_months_by_year)
    most_common_month, most_common_count = month_counts.most_common(1)[0]
    
    # Consistency = % of years peaking in the most common month
    consistency = most_common_count / len(peak_months_by_year)
    
    return round(consistency, 2)

def main():
    conn = get_connection()
    
    # Get keywords that have seasonality data but no consistency score
    keywords_df = pd.read_sql("""
        SELECT DISTINCT KEYWORD 
        FROM SEASONALITY_MONTHLY 
        WHERE PEAK_MONTH IS NOT NULL
    """, conn)
    
    keywords = keywords_df['KEYWORD'].tolist()
    print(f"Found {len(keywords)} keywords to process")
    
    cursor = conn.cursor()
    
    for i, keyword in enumerate(keywords):
        consistency = calculate_peak_consistency(keyword, conn)
        
        if consistency is not None:
            # Update the most recent record for this keyword
            cursor.execute(f"""
                UPDATE SEASONALITY_MONTHLY 
                SET PEAK_CONSISTENCY = {consistency}
                WHERE KEYWORD = '{keyword}'
                AND PEAK_MONTH IS NOT NULL
            """)
            print(f"[{i+1}/{len(keywords)}] {keyword}: {consistency:.0%} consistency")
        else:
            print(f"[{i+1}/{len(keywords)}] {keyword}: no data")
    
    conn.commit()
    conn.close()
    print("Done!")

if __name__ == "__main__":
    main()