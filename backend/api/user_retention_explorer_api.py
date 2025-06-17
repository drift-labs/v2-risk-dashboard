# This new API endpoint will allow for dynamic exploration of user retention for a single market and custom date.

import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Set, Optional, Any

import pandas as pd
from dateutil import tz, parser
from pyathena import connect
import warnings
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import logging
import json

import boto3

def load_markets_from_json(file_path: str) -> Dict[str, Dict[str, Any]]:
    """Loads market data from a JSON file and formats it for the API."""
    try:
        with open(file_path, 'r') as f:
            markets_data = json.load(f)
        
        formatted_markets = {}
        for market in markets_data:
            formatted_markets[market["marketName"]] = {
                "index": market["marketIndex"],
                "launch_ts": market["launchTs"], # Keep original launch_ts for reference if needed
                "category": market["category"]
            }
        logger.info(f"Successfully loaded and formatted {len(formatted_markets)} markets from {file_path}")
        return formatted_markets
    except FileNotFoundError:
        logger.error(f"Market file not found at {file_path}. API will not have market data.")
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

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

ALL_MARKETS = load_markets_from_json("shared/markets.json")

NEW_TRADER_WINDOW_DAYS: int = 7
RETENTION_WINDOWS_DAYS: List[int] = [14, 28]
CHUNK_DAYS: int = 28

DATABASE = os.environ.get("ATHENA_DATABASE", "mainnet-beta-archive")
REGION   = os.environ.get("AWS_REGION", "eu-west-1")
S3_OUTPUT = os.environ.get("ATHENA_S3_OUTPUT", "s3://mainnet-beta-data-ingestion-bucket/athena/")

class RetentionExplorerItem(BaseModel):
    market: str
    category: List[str]
    start_date: str
    new_traders: int
    new_traders_list: List[str]
    retained_users_14d: int
    retention_ratio_14d: float
    retained_users_14d_list: List[str]
    retained_users_28d: int
    retention_ratio_28d: float
    retained_users_28d_list: List[str]

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

def sql_new_traders(mkt_idx: int, start_dt: datetime) -> str:
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
                               mkt_idx: int,
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
          AND  "order".marketindex <> {mkt_idx}
          AND  "user" IN ('{trader_list}')
    '''

async def calculate_retention_for_market(market_name: str, start_date_str: str) -> Dict[str, Any]:
    conn = None
    try:
        start_date = parser.parse(start_date_str).replace(tzinfo=UTC)
        market_config = ALL_MARKETS.get(market_name)
        if not market_config:
            raise HTTPException(status_code=404, detail=f"Market '{market_name}' not found.")

        logger.info(f"Connecting to Athena. S3 staging: {S3_OUTPUT}, Region: {REGION}, DB: {DATABASE}")
        conn = connect(s3_staging_dir=S3_OUTPUT, region_name=REGION, schema_name=DATABASE)
        logger.info("Successfully connected to Athena.")
        log_current_identity()

        # 1. Find new traders for the given market and date
        logger.info(f"Scanning for new traders for {market_name} from {start_date_str}...")
        q_new_traders = sql_new_traders(market_config["index"], start_date)
        new_traders_df = pd.read_sql(q_new_traders, conn)
        logger.info(f"Found {len(new_traders_df)} new traders for {market_name}.")
        
        mkt_traders = new_traders_df["user"].tolist()
        new_traders_count = len(mkt_traders)
        
        result = {
            "market": market_name,
            "category": market_config.get("category", []),
            "start_date": start_date_str,
            "new_traders": new_traders_count,
            "new_traders_list": mkt_traders,
        }

        # 2. Calculate retention for each window
        if not mkt_traders:
             for win in RETENTION_WINDOWS_DAYS:
                result[f"retained_users_{win}d"] = 0
                result[f"retention_ratio_{win}d"] = 0.0
                result[f"retained_users_{win}d_list"] = []
             return result

        retention_period_start_dt = start_date
        for win in RETENTION_WINDOWS_DAYS:
            offset = 0
            retained_set: Set[str] = set()
            
            while offset < win:
                chunk_start_dt = retention_period_start_dt + timedelta(days=offset)
                span = min(CHUNK_DAYS, win - offset)
                if span <= 0: break

                logger.info(f"Fetching retention for {market_name}, window {win}d, chunk: {chunk_start_dt.strftime('%Y-%m-%d')} for {span} days")
                q_retention_chunk = sql_retention_users_chunk(mkt_traders, market_config["index"], chunk_start_dt, span)
                retained_users_df = pd.read_sql(q_retention_chunk, conn)
                retained_set.update(retained_users_df["user"].tolist())
                offset += CHUNK_DAYS

            retained_list = sorted(list(retained_set))
            retained_count = len(retained_list)
            retention_ratio = (retained_count / new_traders_count) if new_traders_count > 0 else 0.0

            result[f"retained_users_{win}d"] = retained_count
            result[f"retention_ratio_{win}d"] = round(retention_ratio, 4)
            result[f"retained_users_{win}d_list"] = retained_list
            
        logger.info(f"Successfully calculated retention for {market_name}.")
        return result

    except Exception as e:
        logger.error(f"Error in calculate_retention_for_market: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process retention data: {str(e)}")
    finally:
        if conn:
            conn.close()
            logger.info("Athena connection closed.")

@router.get("/markets", response_model=List[str])
async def get_available_markets():
    """Returns a list of available market names for the explorer."""
    if not ALL_MARKETS:
        logger.warning("No markets loaded from shared/markets.json")
        return []
    return sorted(list(ALL_MARKETS.keys()))


@router.get("/calculate", response_model=RetentionExplorerItem)
async def get_retention_for_market(
    market_name: str = Query(..., description="The name of the market to analyze."),
    start_date: str = Query(..., description="The start date for the analysis (YYYY-MM-DD).")
):
    """
    Calculates user retention for a specific market from a given start date.
    - Identifies 'new traders' within 7 days of the start date for that market.
    - Measures retention in other markets at 14 and 28 days.
    """
    try:
        logger.info(f"Received request for /calculate: market='{market_name}', date='{start_date}'")
        # Input validation for date format can be added here if needed
        result_data = await calculate_retention_for_market(market_name, start_date)
        return RetentionExplorerItem(**result_data)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Unhandled error in /calculate endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred during calculation.") 