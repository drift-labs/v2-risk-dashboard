import json
import logging
import os
import warnings
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import boto3
import pandas as pd
from dateutil import parser, tz
from fastapi import APIRouter, HTTPException, Query
from pyathena import connect
from pydantic import BaseModel


def load_markets_from_json(file_path: str) -> Dict[str, Dict[str, Any]]:
    """Loads market data from a JSON file and formats it for the API."""
    try:
        with open(file_path, "r") as f:
            markets_data = json.load(f)

        formatted_markets = {}
        for market in markets_data:
            formatted_markets[market["marketName"]] = {
                "index": market["marketIndex"],
                "launch_ts": market[
                    "launchTs"
                ],  # Keep original launch_ts for reference if needed
                "category": market["category"],
            }
        logger.info(
            f"Successfully loaded and formatted {len(formatted_markets)} markets from {file_path}"
        )
        return formatted_markets
    except FileNotFoundError:
        logger.error(
            f"Market file not found at {file_path}. API will not have market data."
        )
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
REGION = os.environ.get("AWS_REGION", "eu-west-1")
S3_OUTPUT = os.environ.get(
    "ATHENA_S3_OUTPUT", "s3://mainnet-beta-data-ingestion-bucket/athena/"
)


class UserVolumeDetails(BaseModel):
    user_address: str
    initial_selected_market_volume: float = 0.0
    initial_other_market_volume: float = 0.0
    selected_market_volume_14d: float = 0.0
    other_market_volume_14d: float = 0.0
    selected_market_volume_28d: float = 0.0
    other_market_volume_28d: float = 0.0


class RetentionExplorerItem(BaseModel):
    market: str
    category: List[str]
    start_date: str
    new_traders_count: int
    retained_14d_count: int
    retained_28d_count: int
    retention_ratio_14d: float
    retention_ratio_28d: float
    user_data: List[UserVolumeDetails]
    total_initial_selected_market_volume_7d: float
    total_initial_other_market_volume_7d: float
    total_volume_14d: float
    total_volume_28d: float


UTC = tz.tzutc()


def dt_from_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1_000, tz=UTC)


def partition_tuples(start: datetime, days: int) -> Set[Tuple[str, str, str]]:
    return {
        (d.strftime("%Y"), d.strftime("%m"), d.strftime("%d"))
        for d in (start + timedelta(n) for n in range(days))
    }


def partition_pred(parts: Set[Tuple[str, str, str]]) -> str:
    lines = [f"(year='{y}' AND month='{m}' AND day='{d}')" for y, m, d in sorted(parts)]
    return " OR ".join(lines)


def sql_new_traders(mkt_idx: int, start_dt: datetime) -> str:
    parts = partition_pred(partition_tuples(start_dt, NEW_TRADER_WINDOW_DAYS))
    return f"""
        WITH potential_new_traders AS (
            -- First, find users who made their first trade in the target market within the window
            SELECT "user", MIN(ts) as first_ts
            FROM eventtype_orderrecord
            WHERE ({parts})
              AND "order".marketindex = {mkt_idx}
            GROUP BY "user"
        ),
        all_trades AS (
            -- Get all trades for these potential new traders across all time
            -- This is needed to ensure the trade in the window was truly their first one
            SELECT "user", MIN(ts) as first_ever_ts
            FROM eventtype_orderrecord
            WHERE "user" IN (SELECT "user" FROM potential_new_traders)
            GROUP BY "user"
        ),
        true_new_traders AS (
            -- Filter to users whose first-ever trade matches their first trade in the window
            SELECT pnt."user"
            FROM potential_new_traders pnt
            JOIN all_trades at ON pnt."user" = at."user" AND pnt.first_ts = at.first_ever_ts
        )
        -- Finally, join with newuserrecord to ensure it's their first subaccount (subaccountid=0)
        SELECT tnt."user"
        FROM true_new_traders tnt
        JOIN eventtype_newuserrecord nur ON tnt."user" = nur."user"
        WHERE nur.subaccountid = 0
"""


# --- New Helper Function for SQL ---
# Define a scaling factor for quote asset amounts, assuming 6 decimal places (e.g., USDC)
QUOTE_ASSET_SCALE_FACTOR = 1e6


def sql_users_volume(
    users: List[str],
    start_dt: datetime,
    end_dt: datetime,
    partition_func: Callable[[Set[Tuple[str, str, str]]], str],
    market_index: Optional[int] = None,
    exclude_market_index: Optional[int] = None,
) -> str:
    """
    Generates SQL to calculate the trading volume for a list of users in a given time period.
    Volume is calculated from `quoteassetamountfilled` in `eventtype_traderecord`.
    """
    if not users:
        return ""

    user_list = "', '".join(users)

    market_filter = ""
    if market_index is not None:
        market_filter = f"AND marketindex = {market_index}"
    if exclude_market_index is not None:
        market_filter = f"AND marketindex <> {exclude_market_index}"

    start_ts = int(start_dt.timestamp())
    # end_dt is exclusive, so we don't subtract one day
    end_ts = int(end_dt.timestamp())

    # Use partitions for traderecord based on the provided function
    num_days = (end_dt - start_dt).days
    parts = partition_func(partition_tuples(start_dt, num_days))

    # This query sums up volume for users as both takers and makers.
    return f"""
        WITH user_trades AS (
            SELECT
                taker AS user,
                CAST(quoteassetamountfilled AS DOUBLE) / {QUOTE_ASSET_SCALE_FACTOR} AS quote_volume
            FROM eventtype_traderecord
            WHERE ({parts}) AND CAST(ts AS BIGINT) >= {start_ts} AND CAST(ts AS BIGINT) < {end_ts}
              AND taker IN ('{user_list}') {market_filter}
            UNION ALL
            SELECT
                maker AS user,
                CAST(quoteassetamountfilled AS DOUBLE) / {QUOTE_ASSET_SCALE_FACTOR} AS quote_volume
            FROM eventtype_traderecord
            WHERE ({parts}) AND CAST(ts AS BIGINT) >= {start_ts} AND CAST(ts AS BIGINT) < {end_ts}
              AND maker IN ('{user_list}') {market_filter}
        )
        SELECT
            user,
            SUM(quote_volume) AS total_volume
        FROM user_trades
        GROUP BY user
    """


async def calculate_retention_for_market(
    market_name: str, start_date_str: str
) -> Dict[str, Any]:
    conn = None
    try:
        start_date = parser.parse(start_date_str).replace(tzinfo=UTC)
        market_config = ALL_MARKETS.get(market_name)
        if not market_config:
            raise HTTPException(
                status_code=404, detail=f"Market '{market_name}' not found."
            )

        mkt_idx = market_config["index"]

        logger.info(
            f"Connecting to Athena. S3 staging: {S3_OUTPUT}, Region: {REGION}, DB: {DATABASE}"
        )
        conn = connect(
            s3_staging_dir=S3_OUTPUT, region_name=REGION, schema_name=DATABASE
        )
        logger.info("Successfully connected to Athena.")
        log_current_identity()

        # 1. Find new traders
        logger.info(
            f"Scanning for new traders for {market_name} from {start_date_str}..."
        )
        q_new_traders = sql_new_traders(mkt_idx, start_date)
        new_traders_df = pd.read_sql(q_new_traders, conn)
        mkt_traders = new_traders_df["user"].tolist()
        new_traders_count = len(mkt_traders)

        if not mkt_traders:
            return {
                "market": market_name,
                "category": market_config.get("category", []),
                "start_date": start_date_str,
                "new_traders_count": 0,
                "retained_14d_count": 0,
                "retained_28d_count": 0,
                "retention_ratio_14d": 0.0,
                "retention_ratio_28d": 0.0,
                "user_data": [],
                "total_initial_selected_market_volume_7d": 0.0,
                "total_initial_other_market_volume_7d": 0.0,
                "total_volume_14d": 0.0,
                "total_volume_28d": 0.0,
            }

        # --- Volume Calculation ---
        traders_df = pd.DataFrame({"user": mkt_traders})

        # Define time windows
        initial_window_end = start_date + timedelta(days=NEW_TRADER_WINDOW_DAYS)
        retention_14d_end = initial_window_end + timedelta(days=14)
        retention_28d_end = initial_window_end + timedelta(days=28)

        def get_and_merge_volume(
            base_df: pd.DataFrame,
            column_name: str,
            start_dt: datetime,
            end_dt: datetime,
            m_idx: Optional[int] = None,
            exclude_m_idx: Optional[int] = None,
        ) -> pd.DataFrame:
            logger.info(f"Calculating volume for '{column_name}'...")
            vol_sql = sql_users_volume(
                mkt_traders, start_dt, end_dt, partition_pred, m_idx, exclude_m_idx
            )
            if not vol_sql:
                return base_df.assign(**{column_name: 0.0})

            vol_df = pd.read_sql(vol_sql, conn)
            merged_df = base_df.merge(vol_df, on="user", how="left")

            # Explicitly convert to numeric to avoid FutureWarning on downcasting
            merged_df["total_volume"] = pd.to_numeric(
                merged_df["total_volume"], errors="coerce"
            )
            merged_df["total_volume"] = merged_df["total_volume"].fillna(0)
            return merged_df.rename(columns={"total_volume": column_name})

        # Calculate all volume metrics
        traders_df = get_and_merge_volume(
            traders_df,
            "initial_selected_market_volume",
            start_date,
            initial_window_end,
            m_idx=mkt_idx,
        )
        traders_df = get_and_merge_volume(
            traders_df,
            "initial_other_market_volume",
            start_date,
            initial_window_end,
            exclude_m_idx=mkt_idx,
        )
        traders_df = get_and_merge_volume(
            traders_df,
            "selected_market_volume_14d",
            initial_window_end,
            retention_14d_end,
            m_idx=mkt_idx,
        )
        traders_df = get_and_merge_volume(
            traders_df,
            "other_market_volume_14d",
            initial_window_end,
            retention_14d_end,
            exclude_m_idx=mkt_idx,
        )
        traders_df = get_and_merge_volume(
            traders_df,
            "selected_market_volume_28d",
            initial_window_end,
            retention_28d_end,
            m_idx=mkt_idx,
        )
        traders_df = get_and_merge_volume(
            traders_df,
            "other_market_volume_28d",
            initial_window_end,
            retention_28d_end,
            exclude_m_idx=mkt_idx,
        )

        traders_df = traders_df.rename(columns={"user": "user_address"})

        # --- Summary Volume Metrics ---
        total_initial_selected_market_volume_7d = traders_df[
            "initial_selected_market_volume"
        ].sum()
        total_initial_other_market_volume_7d = traders_df[
            "initial_other_market_volume"
        ].sum()
        total_volume_14d = (
            traders_df["selected_market_volume_14d"]
            + traders_df["other_market_volume_14d"]
        ).sum()
        total_volume_28d = (
            traders_df["selected_market_volume_28d"]
            + traders_df["other_market_volume_28d"]
        ).sum()

        # --- Summary Statistics ---
        retained_14d_count = int((traders_df["other_market_volume_14d"] > 0).sum())
        retained_28d_count = int((traders_df["other_market_volume_28d"] > 0).sum())
        retention_ratio_14d = (
            (retained_14d_count / new_traders_count) if new_traders_count > 0 else 0.0
        )
        retention_ratio_28d = (
            (retained_28d_count / new_traders_count) if new_traders_count > 0 else 0.0
        )

        result = {
            "market": market_name,
            "category": market_config.get("category", []),
            "start_date": start_date_str,
            "new_traders_count": new_traders_count,
            "retained_14d_count": retained_14d_count,
            "retained_28d_count": retained_28d_count,
            "retention_ratio_14d": round(retention_ratio_14d, 4),
            "retention_ratio_28d": round(retention_ratio_28d, 4),
            "user_data": traders_df.to_dict("records"),
            "total_initial_selected_market_volume_7d": float(
                total_initial_selected_market_volume_7d
            ),
            "total_initial_other_market_volume_7d": float(
                total_initial_other_market_volume_7d
            ),
            "total_volume_14d": float(total_volume_14d),
            "total_volume_28d": float(total_volume_28d),
        }

        logger.info(
            f"Successfully calculated consolidated retention for {market_name}."
        )
        return result

    except Exception as e:
        logger.error(f"Error in calculate_retention_for_market: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to process retention data: {str(e)}"
        )
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
    start_date: str = Query(
        ..., description="The start date for the analysis (YYYY-MM-DD)."
    ),
):
    """
    Calculates user retention for a specific market from a given start date.
    - Identifies 'new traders' within 7 days of the start date for that market.
    - Measures retention in other markets at 14 and 28 days.
    """
    try:
        logger.info(
            f"Received request for /calculate: market='{market_name}', date='{start_date}'"
        )
        result_data = await calculate_retention_for_market(market_name, start_date)
        return RetentionExplorerItem(**result_data)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Unhandled error in /calculate endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal server error occurred during calculation.",
        )
