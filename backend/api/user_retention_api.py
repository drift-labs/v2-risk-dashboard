# This new API endpoint will implement hype_market_retention.py and return it as a JSON object.

import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Set, Optional, Any

import pandas as pd
from dateutil import tz
from pyathena import connect
import warnings
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging

import boto3

def log_current_identity():
    try:
        sts = boto3.client("sts")
        identity = sts.get_caller_identity()
        logger.info(f"Running as: {identity}")
    except Exception as e:
        logger.warning(f"Could not determine AWS identity: {e}")

# Ignore UserWarning from pyathena/pandas if any
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# ───────────────────────── 1. CONFIG (adapted from hype_market_retention.py) ───────────────────────── #

HYPE_MARKETS: Dict[str, Dict[str, int]] = {
    "WIF-PERP":   {"index": 23, "launch_ts": 1706219971000},
    "POPCAT-PERP":{"index": 34, "launch_ts": 1720013054000},
    "HYPE-PERP":  {"index": 59, "launch_ts": 1733374800000}, # Example, adjust as needed
    "KAITO-PERP": {"index": 69, "launch_ts": 1739545901000}, # Example, adjust as needed
    "FARTCOIN-PERP": {"index": 71, "launch_ts": 1743086746000}, # Example, adjust as needed
    "TRUMP-PERP": {"index": 64, "launch_ts": 1737219250000}, # Example, adjust as needed
    "LAUNCHCOIN-PERP": {"index": 74, "launch_ts": 1747318237000} # Example, adjust as needed
}

NEW_TRADER_WINDOW_DAYS: int = 7
RETENTION_WINDOWS_DAYS: List[int] = [14, 28] # Affects Pydantic model if changed
CHUNK_DAYS: int = 28

DATABASE = os.environ.get("ATHENA_DATABASE", "mainnet-beta-archive")
REGION   = os.environ.get("AWS_REGION", "eu-west-1")
S3_OUTPUT = os.environ.get("ATHENA_S3_OUTPUT", "s3://mainnet-beta-data-ingestion-bucket/athena/")


# Derived constant for easier lookup by market index
# MARKET_CONFIG_BY_INDEX: Dict[int, Dict[str, any]] = {
# cfg["index"]: {"name": name, **cfg} for name, cfg in HYPE_MARKETS.items()
# }

# ──────────────────────── 1A. Pydantic Models ──────────────────────── #

class RetentionSummaryItem(BaseModel):
    market: str
    new_traders: int
    new_traders_list: Optional[List[str]] = None
    retained_users_14d: Optional[int] = None
    retention_ratio_14d: Optional[float] = None
    retained_users_14d_list: Optional[List[str]] = None
    retained_users_28d: Optional[int] = None
    retention_ratio_28d: Optional[float] = None
    retained_users_28d_list: Optional[List[str]] = None

    class Config:
        orm_mode = True # Changed from allow_population_by_field_name for Pydantic v2
        # For Pydantic v1, it would be:
        # allow_population_by_field_name = True 
        # To handle field names like 'retained_users_14d' if data comes with underscores


# ──────────────────────── 2. HELPERS (Copied and adapted from hype_market_retention.py) ──────────────────────── #

UTC = tz.tzutc()

def dt_from_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1_000, tz=UTC)

def partition_tuples(start: datetime, days: int) -> Set[Tuple[str, str, str]]:
    return {
        (d.strftime("%Y"), d.strftime("%m"), d.strftime("%d"))
        for d in (start + timedelta(n) for n in range(days))
    }

def partition_pred(parts: Set[Tuple[str, str, str]]) -> str:
    lines = [
        f"(year='{y}' AND month='{m}' AND day='{d}')" for y, m, d in sorted(parts)
    ]
    return " OR ".join(lines)

def sql_new_traders(mkt_idx: int, launch_ms: int) -> str:
    start_dt = dt_from_ms(launch_ms)
    parts = partition_pred(partition_tuples(start_dt, NEW_TRADER_WINDOW_DAYS))
    return f"""
        SELECT "user",
               MIN(slot) AS first_slot,
               MIN(ts)   AS first_ts
        FROM   eventtype_orderrecord
        WHERE  ({parts})
          AND  "order".marketindex = {mkt_idx}
          AND  ("order".orderid = 0 OR "order".orderid = 1)
        GROUP  BY "user"
    """

def sql_retention_users_chunk(traders: List[str],
                               hype_idx: int,
                               chunk_start: datetime,
                               chunk_days: int) -> str:
    chunk_end = chunk_start + timedelta(days=chunk_days)
    start_ts = int(chunk_start.timestamp())
    end_ts = int(chunk_end.timestamp())
    from_date = chunk_start.strftime('%Y%m%d')
    to_date = chunk_end.strftime('%Y%m%d')
    trader_list = "', '".join(traders)

    return f'''
        WITH time_range AS (
            SELECT 
                {start_ts} AS from_ts,
                {end_ts} AS to_ts,
                '{from_date}' AS from_date,
                '{to_date}' AS to_date
        )
        SELECT DISTINCT "user"
        FROM   eventtype_orderrecord, time_range
        WHERE  CAST(ts AS INT) BETWEEN time_range.from_ts AND time_range.to_ts
          AND  CONCAT(year, month, day) BETWEEN time_range.from_date AND time_range.to_date
          AND  "order".marketindex <> {hype_idx}
          AND  "user" IN ('{trader_list}')
    '''

# ───────────────────── 3. MAIN DATA FETCHING AND PROCESSING LOGIC ───────────────────── #

async def fetch_and_process_retention_data() -> pd.DataFrame:
    conn = None
    try:
        logger.info(f"Connecting to Athena. S3 staging: {S3_OUTPUT}, Region: {REGION}, DB: {DATABASE}")
        conn = connect(
            s3_staging_dir=S3_OUTPUT,
            region_name=REGION,
            schema_name=DATABASE, # schema_name is used for the default database in queries
        )
        logger.info("Successfully connected to Athena.")
        
        log_current_identity()

        # 3A. discover "new traders" per market
        logger.info("Scanning for new traders across all hype markets...")
        new_traders_frames: List[pd.DataFrame] = []
        for mkt, cfg in HYPE_MARKETS.items():
            logger.info(f"• Scanning new traders for {mkt}…")
            q = sql_new_traders(cfg["index"], cfg["launch_ts"])
            # logger.debug(f"Generated SQL for {mkt} new traders:\\n{q}")
            try:
                df = pd.read_sql(q, conn)
                df["market"] = mkt
                new_traders_frames.append(df)
                logger.info(f"Found {len(df)} new traders for {mkt}.")
            except Exception as e:
                logger.error(f"Error fetching new traders for market {mkt}: {e}")
                # Optionally, append an empty DataFrame or specific error marker
                # For now, just logs and continues, potentially resulting in empty new_traders for this market
        
        if not new_traders_frames:
            logger.warning("No new traders found for any market.")
            new_traders = pd.DataFrame(columns=["trader", "first_slot", "first_ts", "market"])
        else:
            new_traders = (
                pd.concat(new_traders_frames, ignore_index=True)
                  .rename(columns={"user": "trader"})
            )
        logger.info(f"Processed a total of {len(new_traders)} new trader records across all markets.")

        # 3AA. Prepare list of new traders per market
        if not new_traders.empty:
            new_traders_lists_df = new_traders.groupby('market')['trader'].apply(list).reset_index(name='new_traders_list')
        else:
            new_traders_lists_df = pd.DataFrame({'market': pd.Series(dtype='str'), 'new_traders_list': pd.Series(dtype='object')})

        # 3B. retention look-ups
        retention_records: List[Dict[str, object]] = []
        if not new_traders.empty:
            for mkt, cfg in HYPE_MARKETS.items():
                mkt_traders = new_traders[new_traders.market == mkt].trader.tolist()
                if not mkt_traders:
                    logger.info(f"No new traders for market {mkt}, skipping retention lookup.")
                    continue

                hype_idx = cfg["index"]
                launch_ms = cfg["launch_ts"]
                # The original script uses `start_dt = dt_from_ms(launch_ms)` here.
                # This `start_dt` is the launch time of the hype market.
                # Retention is checked *from this launch time* up to `win` days.
                retention_period_start_dt = dt_from_ms(launch_ms) 

                for win in RETENTION_WINDOWS_DAYS:
                    offset = 0
                    retained_set: Set[str] = set()
                    
                    # The retention window is [launch_ts, launch_ts + win days).
                    # We query in chunks within this period.
                    while offset < win : # Iterate through chunks covering the *win* day retention period
                        # Chunk starts from retention_period_start_dt + offset
                        chunk_start_dt = retention_period_start_dt + timedelta(days=offset)
                        # Span is CHUNK_DAYS, but not exceeding the total 'win' days from retention_period_start_dt
                        # Max days for this chunk from chunk_start_dt: win - offset
                        span = min(CHUNK_DAYS, win - offset)
                        
                        if span <= 0: # Should not happen if while condition is offset < win
                            break

                        logger.info(f"Fetching retention for {mkt}, window {win}d, chunk: {chunk_start_dt.strftime('%Y-%m-%d')} for {span} days, {len(mkt_traders)} traders")
                        q_retention_chunk = sql_retention_users_chunk(mkt_traders, hype_idx, chunk_start_dt, span)
                        # logger.debug(f"Retention SQL for chunk:\n{q_retention_chunk}")
                        try:
                            retained_users_df = pd.read_sql(q_retention_chunk, conn)
                            retained_set.update(retained_users_df["user"].tolist())
                        except Exception as e: # Catch issues with individual SQL queries
                             logger.error(f"Error fetching retention chunk for {mkt}, window {win}d, chunk {offset}: {e}")
                        offset += CHUNK_DAYS

                    for trader in mkt_traders:
                        retention_records.append({
                            "market": mkt,
                            "trader": trader,
                            "window_days": win,
                            "retained": trader in retained_set
                        })
        else:
            logger.info("Skipping retention lookups as no new traders were found.")


        if not retention_records:
            logger.warning("No retention records generated. Summary will be empty or based on new traders only if any.")
            # If new_traders is not empty, but retention_records is, means no one was retained.
            # The aggregation logic below should handle this by creating 0s for retained counts/ratios.
            # Initialize with expected columns if needed for robust aggregation, though pandas handles it.
            retention = pd.DataFrame(columns=["market", "trader", "window_days", "retained"])
        else:
            retention = pd.DataFrame(retention_records)
        
        logger.info(f"Generated {len(retention)} retention records.")

        # 3BB. Prepare lists of retained traders per market and window
        if not retention.empty and 'retained' in retention.columns and 'trader' in retention.columns:
            retained_traders_lists_df = retention[retention['retained'] == True].groupby(['market', 'window_days'])['trader'].apply(list).reset_index(name='retained_traders_list')
        else:
            retained_traders_lists_df = pd.DataFrame({
                'market': pd.Series(dtype='str'), 
                'window_days': pd.Series(dtype='int'), 
                'retained_traders_list': pd.Series(dtype='object')
            })


        # 4. aggregate metrics
        if retention.empty and new_traders.empty:
            logger.info("Both new_traders and retention are empty. Returning empty summary.")
            # Ensure all expected columns exist, even if with no data, matching Pydantic model
            cols = ['market', 'new_traders', 'new_traders_list']
            for win_days in RETENTION_WINDOWS_DAYS:
                cols.append(f'retained_users_{win_days}d')
                cols.append(f'retention_ratio_{win_days}d')
                cols.append(f'retained_users_{win_days}d_list')
            return pd.DataFrame(columns=cols)

        summary_intermediate = pd.DataFrame()
        if not retention.empty:
            summary_intermediate = (
                retention.groupby(["market", "window_days"])
                         .agg(new_traders_count_in_group=("retained", "size"),
                              retained_users_sum_in_group=("retained", "sum"))
                         .reset_index()
            )
            if not summary_intermediate.empty: # Avoid division by zero if no groups or counts are zero
                 summary_intermediate["retention_ratio_for_group"] = (
                    summary_intermediate.retained_users_sum_in_group.astype(float) / 
                    summary_intermediate.new_traders_count_in_group.astype(float)
                ).round(3)
            else: # Handle case where retention is not empty but groupby results in empty intermediate
                summary_intermediate["retention_ratio_for_group"] = pd.Series(dtype=float)


        # Initialize final_summary with total new traders per market from `new_traders` DataFrame
        if not new_traders.empty:
            final_summary_counts = new_traders.groupby('market').size().reset_index(name='new_traders')
        else: # No new traders found at all
            final_summary_counts = pd.DataFrame({'market': pd.Series(dtype='str'), 'new_traders': pd.Series(dtype='int')})

        # Ensure all HYPE_MARKETS are present in final_summary, even if they had 0 new traders
        all_market_names_df = pd.DataFrame({'market': list(HYPE_MARKETS.keys())})
        final_summary = pd.merge(all_market_names_df, final_summary_counts, on='market', how='left').fillna({'new_traders': 0})
        final_summary['new_traders'] = final_summary['new_traders'].astype(int)
        
        # Merge new traders lists
        final_summary = pd.merge(final_summary, new_traders_lists_df, on='market', how='left')
        # Ensure 'new_traders_list' column exists and fill NaNs with empty lists
        if 'new_traders_list' not in final_summary.columns:
            final_summary['new_traders_list'] = [[] for _ in range(len(final_summary))]
        else:
            final_summary['new_traders_list'] = final_summary['new_traders_list'].apply(lambda d: d if isinstance(d, list) else [])
        
        final_summary = final_summary.set_index('market')


        for win_days in RETENTION_WINDOWS_DAYS:
            retained_count_col = f'retained_users_{win_days}d'
            retention_ratio_col = f'retention_ratio_{win_days}d'
            retained_list_col = f'retained_users_{win_days}d_list'
            
            if not summary_intermediate.empty:
                window_data = summary_intermediate[summary_intermediate.window_days == win_days]
                if not window_data.empty:
                    window_metrics = window_data[['market', 'retained_users_sum_in_group', 'retention_ratio_for_group']].set_index('market')
                    window_metrics = window_metrics.rename(columns={
                        'retained_users_sum_in_group': retained_count_col,
                        'retention_ratio_for_group': retention_ratio_col
                    })
                    final_summary = final_summary.join(window_metrics)
            
            # Join retained traders lists for this window
            if not retained_traders_lists_df.empty:
                window_list_data = retained_traders_lists_df[retained_traders_lists_df.window_days == win_days]
                if not window_list_data.empty:
                    window_list_metrics = window_list_data[['market', 'retained_traders_list']].set_index('market')
                    window_list_metrics = window_list_metrics.rename(columns={'retained_traders_list': retained_list_col})
                    final_summary = final_summary.join(window_list_metrics)

            # Fill NaN values for this window's columns (counts, ratios, and lists)
            final_summary[retained_count_col] = final_summary.get(retained_count_col, pd.Series(0, index=final_summary.index)).fillna(0).astype(int)
            final_summary[retention_ratio_col] = final_summary.get(retention_ratio_col, pd.Series(0.0, index=final_summary.index)).fillna(0.0).astype(float)
            # For list columns, ensure they exist and NaNs are filled with empty lists
            if retained_list_col not in final_summary.columns:
                 final_summary[retained_list_col] = pd.Series([[] for _ in range(len(final_summary))], index=final_summary.index, dtype='object')
            else:
                final_summary[retained_list_col] = final_summary[retained_list_col].apply(lambda d: d if isinstance(d, list) else [])
        
        final_summary = final_summary.reset_index()
        logger.info("Successfully generated retention summary.")
        # logger.debug(f"Final Summary:\n{final_summary.to_string()}")
        return final_summary

    except Exception as e:
        logger.error(f"Error in fetch_and_process_retention_data: {e}", exc_info=True)
        # For Pydantic model compatibility, return DataFrame with expected columns in case of error
        cols = ['market', 'new_traders', 'new_traders_list']
        for w in RETENTION_WINDOWS_DAYS:
            cols.extend([f'retained_users_{w}d', f'retention_ratio_{w}d', f'retained_users_{w}d_list'])
        empty_df_on_error = pd.DataFrame(columns=cols)
        
        error_data = []
        for market_name in HYPE_MARKETS.keys():
            record: Dict[str, Any] = {'market': market_name, 'new_traders': 0, 'new_traders_list': []}
            for w in RETENTION_WINDOWS_DAYS:
                record[f'retained_users_{w}d'] = 0
                record[f'retention_ratio_{w}d'] = 0.0
                record[f'retained_users_{w}d_list'] = []
            error_data.append(record)
        if error_data:
            empty_df_on_error = pd.DataFrame(error_data)
        # Instead of returning, we raise HTTPException, so this df is for consistency if we changed to return it.
        # However, the API will return based on the exception.
        # For clarity, let's ensure the raised exception occurs rather than returning a df from here.
        raise HTTPException(status_code=500, detail=f"Failed to fetch or process user retention data: {str(e)}")

    finally:
        if conn:
            conn.close()
            logger.info("Athena connection closed.")

# ───────────────────────── 4. API Endpoint ───────────────────────── #

@router.get("/summary", response_model=List[RetentionSummaryItem])
async def get_user_retention_summary():
    """
    Provides a summary of user retention for "hype" markets.
    - Identifies traders whose first order was in a hype market shortly after launch.
    - Measures how many of those traders were active in *other* markets within 14 and 28 days.
    """
    try:
        logger.info("Received request for /summary endpoint.")
        summary_df = await fetch_and_process_retention_data()
        
        if summary_df.empty and not HYPE_MARKETS: # No markets configured
             return []
        if summary_df.empty and HYPE_MARKETS: # Markets configured but no data (e.g. no new users at all)
            # Construct a list of RetentionSummaryItem with 0 values for all configured markets
            # This ensures the frontend gets a consistent structure.
            results = []
            for market_name in HYPE_MARKETS.keys():
                item_data: Dict[str, Any] = {"market": market_name, "new_traders": 0, "new_traders_list": []}
                for win_day in RETENTION_WINDOWS_DAYS:
                    item_data[f"retained_users_{win_day}d"] = 0
                    item_data[f"retention_ratio_{win_day}d"] = 0.0
                    item_data[f"retained_users_{win_day}d_list"] = []
                results.append(RetentionSummaryItem(**item_data))
            return results

        # Convert DataFrame to list of dicts for Pydantic model validation
        results_dict = summary_df.to_dict(orient='records')
        
        # Validate with Pydantic model
        validated_results = [RetentionSummaryItem(**item) for item in results_dict]
        logger.info(f"Successfully processed request for /summary. Returning {len(validated_results)} summary items.")
        return validated_results
    except HTTPException as http_exc:
        # Re-raise HTTPExceptions directly
        raise http_exc
    except Exception as e:
        logger.error(f"Unhandled error in /summary endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

# To run this API locally (example):
# Ensure FastAPI and Uvicorn are installed: pip install fastapi uvicorn
# Save this file as user_retention_api.py
# Run with: uvicorn user_retention_api:router --reload --port 8001 (assuming this router is added to a FastAPI app)