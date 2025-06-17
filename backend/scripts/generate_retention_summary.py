# This script generates a static user_retention_summary.json file.
# It contains the original data fetching and processing logic from the initial implementation
# of the user_retention_summary_api.py. Its purpose is to be run once to create the
# static data file that the refactored API will serve.

import os
import sys
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Set, Any

import pandas as pd
from dateutil import tz
from pyathena import connect
import warnings
import logging

import boto3

def load_markets_from_json(file_path: str) -> Dict[str, Dict[str, Any]]:
    """Loads market data from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            markets_data = json.load(f)
        
        formatted_markets = {}
        for market in markets_data:
            formatted_markets[market["marketName"]] = {
                "index": market["marketIndex"],
                "launch_ts": market["launchTs"],
                "category": market["category"]
            }
        logger.info(f"Successfully loaded {len(formatted_markets)} markets from {file_path}")
        return formatted_markets
    except FileNotFoundError:
        logger.error(f"Market file not found at {file_path}.")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {file_path}.")
        return {}
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading markets: {e}")
        return {}

def log_current_identity():
    try:
        sts = boto3.client("sts")
        identity = sts.get_caller_identity()
        logger.info(f"Running as: {identity}")
    except Exception as e:
        logger.warning(f"Could not determine AWS identity: {e}")

# Add project root to sys.path to allow imports from shared
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Ignore UserWarning from pyathena/pandas
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIG ---
ALL_MARKETS = load_markets_from_json("shared/markets.json")
OUTPUT_FILE_PATH = "shared/user_retention_summary.json"

NEW_TRADER_WINDOW_DAYS: int = 7
RETENTION_WINDOWS_DAYS: List[int] = [14, 28]
CHUNK_DAYS: int = 28

DATABASE = os.environ.get("ATHENA_DATABASE", "mainnet-beta-archive")
REGION   = os.environ.get("AWS_REGION", "eu-west-1")
S3_OUTPUT = os.environ.get("ATHENA_S3_OUTPUT", "s3://mainnet-beta-data-ingestion-bucket/athena/")

# --- HELPERS ---
UTC = tz.tzutc()

def dt_from_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1_000, tz=UTC)

def partition_tuples(start: datetime, days: int) -> Set[Tuple[str, str, str]]:
    return {(d.strftime("%Y"), d.strftime("%m"), d.strftime("%d")) for d in (start + timedelta(n) for n in range(days))}

def partition_pred(parts: Set[Tuple[str, str, str]]) -> str:
    lines = [f"(year='{y}' AND month='{m}' AND day='{d}')" for y, m, d in sorted(parts)]
    return " OR ".join(lines)

def sql_new_traders(mkt_idx: int, launch_ms: int) -> str:
    start_dt = dt_from_ms(launch_ms)
    parts = partition_pred(partition_tuples(start_dt, NEW_TRADER_WINDOW_DAYS))
    return f"""
        SELECT "user", MIN(slot) AS first_slot, MIN(ts) AS first_ts
        FROM eventtype_orderrecord
        WHERE ({parts}) AND "order".marketindex = {mkt_idx} AND ("order".orderid = 0 OR "order".orderid = 1)
        GROUP BY "user"
    """

def sql_retention_users_chunk(traders: List[str], hype_idx: int, chunk_start: datetime, chunk_days: int) -> str:
    chunk_end = chunk_start + timedelta(days=chunk_days)
    start_ts, end_ts = int(chunk_start.timestamp()), int(chunk_end.timestamp())
    from_date, to_date = chunk_start.strftime('%Y%m%d'), chunk_end.strftime('%Y%m%d')
    trader_list = "', '".join(traders)
    return f"""
        WITH time_range AS (SELECT {start_ts} AS from_ts, {end_ts} AS to_ts, '{from_date}' AS from_date, '{to_date}' AS to_date)
        SELECT DISTINCT "user"
        FROM eventtype_orderrecord, time_range
        WHERE CAST(ts AS INT) BETWEEN time_range.from_ts AND time_range.to_ts
          AND CONCAT(year, month, day) BETWEEN time_range.from_date AND time_range.to_date
          AND "order".marketindex <> {hype_idx} AND "user" IN ('{trader_list}')
    """

# --- MAIN LOGIC ---
async def generate_summary_data() -> None:
    conn = None
    try:
        logger.info(f"Connecting to Athena. S3 staging: {S3_OUTPUT}, Region: {REGION}, DB: {DATABASE}")
        conn = connect(s3_staging_dir=S3_OUTPUT, region_name=REGION, schema_name=DATABASE)
        logger.info("Successfully connected to Athena.")
        log_current_identity()

        # Step 1: Discover "new traders" per market
        logger.info("Scanning for new traders across all markets...")
        new_traders_frames: List[pd.DataFrame] = []
        for mkt, cfg in ALL_MARKETS.items():
            logger.info(f"• Scanning new traders for {mkt}…")
            q = sql_new_traders(cfg["index"], cfg["launch_ts"])
            try:
                df = pd.read_sql(q, conn)
                df["market"] = mkt
                new_traders_frames.append(df)
                logger.info(f"Found {len(df)} new traders for {mkt}.")
            except Exception as e:
                logger.error(f"Error fetching new traders for market {mkt}: {e}")

        new_traders = pd.concat(new_traders_frames, ignore_index=True).rename(columns={"user": "trader"}) if new_traders_frames else pd.DataFrame(columns=["trader", "market"])

        # Step 2: Perform retention look-ups
        retention_records: List[Dict[str, Any]] = []
        if not new_traders.empty:
            for mkt, cfg in ALL_MARKETS.items():
                mkt_traders = new_traders[new_traders.market == mkt].trader.tolist()
                if not mkt_traders: continue
                
                retention_period_start_dt = dt_from_ms(cfg["launch_ts"])
                for win in RETENTION_WINDOWS_DAYS:
                    retained_set: Set[str] = set()
                    offset = 0
                    while offset < win:
                        chunk_start_dt = retention_period_start_dt + timedelta(days=offset)
                        span = min(CHUNK_DAYS, win - offset)
                        if span <= 0: break
                        
                        logger.info(f"Fetching retention for {mkt}, window {win}d, chunk: {chunk_start_dt.strftime('%Y-%m-%d')} for {span} days")
                        q_retention = sql_retention_users_chunk(mkt_traders, cfg["index"], chunk_start_dt, span)
                        retained_users_df = pd.read_sql(q_retention, conn)
                        retained_set.update(retained_users_df["user"].tolist())
                        offset += CHUNK_DAYS

                    for trader in mkt_traders:
                        retention_records.append({"market": mkt, "trader": trader, "window_days": win, "retained": trader in retained_set})

        retention = pd.DataFrame(retention_records) if retention_records else pd.DataFrame(columns=["market", "trader", "window_days", "retained"])

        # Step 3: Aggregate metrics and build final summary DataFrame
        logger.info("Aggregating final metrics...")
        all_markets_df = pd.DataFrame([{'market': name, 'category': config['category']} for name, config in ALL_MARKETS.items()])
        
        if new_traders.empty:
            final_summary = all_markets_df.copy()
            final_summary['new_traders'] = 0
        else:
            final_summary_counts = new_traders.groupby('market').size().reset_index(name='new_traders')
            final_summary = pd.merge(all_markets_df, final_summary_counts, on='market', how='left').fillna({'new_traders': 0})
        final_summary['new_traders'] = final_summary['new_traders'].astype(int)

        if not new_traders.empty:
            new_traders_lists_df = new_traders.groupby('market')['trader'].apply(list).reset_index(name='new_traders_list')
            final_summary = pd.merge(final_summary, new_traders_lists_df, on='market', how='left')
        
        final_summary['new_traders_list'] = final_summary['new_traders_list'].apply(lambda d: d if isinstance(d, list) else [])

        final_summary = final_summary.set_index('market')
        
        if not retention.empty:
            summary_intermediate = retention.groupby(["market", "window_days"]).agg(retained_users_sum=("retained", "sum")).reset_index()
            retained_traders_lists_df = retention[retention['retained']].groupby(['market', 'window_days'])['trader'].apply(list).reset_index(name='retained_traders_list')

            for win_days in RETENTION_WINDOWS_DAYS:
                retained_count_col = f'retained_users_{win_days}d'
                retention_ratio_col = f'retention_ratio_{win_days}d'
                retained_list_col = f'retained_users_{win_days}d_list'

                window_data = summary_intermediate[summary_intermediate.window_days == win_days].set_index('market')
                final_summary[retained_count_col] = window_data['retained_users_sum']
                
                window_list_data = retained_traders_lists_df[retained_traders_lists_df.window_days == win_days]
                if not window_list_data.empty:
                    window_list_metrics = window_list_data[['market', 'retained_traders_list']].set_index('market')
                    window_list_metrics = window_list_metrics.rename(columns={'retained_traders_list': retained_list_col})
                    final_summary = final_summary.join(window_list_metrics)
        
        # Post-processing and filling NaNs
        for win_days in RETENTION_WINDOWS_DAYS:
            count_col = f'retained_users_{win_days}d'
            ratio_col = f'retention_ratio_{win_days}d'
            list_col = f'retained_users_{win_days}d_list'
            
            final_summary[count_col] = final_summary.get(count_col, 0).fillna(0).astype(int)
            final_summary[ratio_col] = (final_summary[count_col] / final_summary['new_traders'].replace(0, 1)).fillna(0).astype(float) # Avoid division by zero
            if list_col not in final_summary.columns:
                final_summary[list_col] = pd.Series([[] for _ in range(len(final_summary))], index=final_summary.index)
            final_summary[list_col] = final_summary[list_col].apply(lambda d: d if isinstance(d, list) else [])

        final_summary = final_summary.reset_index()

        # Step 4: Save to JSON
        logger.info(f"Saving summary data to {OUTPUT_FILE_PATH}...")
        output_data = final_summary.to_dict(orient='records')
        
        os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
        with open(OUTPUT_FILE_PATH, 'w') as f:
            json.dump(output_data, f, indent=4)
        
        logger.info("Successfully generated and saved user retention summary.")

    except Exception as e:
        logger.error(f"An error occurred during summary generation: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()
            logger.info("Athena connection closed.")

if __name__ == "__main__":
    # Ensure environment variables are set (e.g., from a .env file or manually)
    if "AWS_REGION" not in os.environ:
        logger.warning("AWS_REGION environment variable not set. Defaulting to eu-west-1, but this may fail.")
    
    logger.info("Starting user retention summary generation script.")
    asyncio.run(generate_summary_data())
    logger.info("Script finished.") 