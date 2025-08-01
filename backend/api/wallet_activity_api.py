import logging
import os
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Set, Tuple

import pandas as pd
from dateutil import parser, tz
from fastapi import APIRouter, HTTPException, Query
from pyathena import connect

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()

DATABASE = os.environ.get("ATHENA_DATABASE", "mainnet-beta-archive")
REGION = os.environ.get("AWS_REGION", "eu-west-1")
S3_OUTPUT = os.environ.get(
    "ATHENA_S3_OUTPUT", "s3://mainnet-beta-data-ingestion-bucket/athena/"
)

QUOTE_ASSET_SCALE_FACTOR = 1e6
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


def sql_get_all_traders_since_date(start_date: datetime) -> str:
    """Get all unique traders who have traded since the start date."""
    now = datetime.now(tz=UTC)
    parts = partition_pred(partition_tuples(start_date, (now - start_date).days + 1))
    start_ts = int(start_date.timestamp())

    full_query = f"""
        SELECT DISTINCT "user"
        FROM eventtype_orderrecord
        WHERE ({parts}) AND CAST(ts AS BIGINT) >= {start_ts}
    """

    print(f"Sql get all traders since date:\n {full_query}")
    return full_query


def sql_get_user_first_trade_date(users: List[str]) -> str:
    """Get the first trade date for each user."""
    if not users:
        return ""
    user_list = "', '".join(users)

    full_query = f"""
        SELECT 
            "user",
            MIN(CAST(ts AS BIGINT)) as first_trade_ts
        FROM eventtype_orderrecord
        WHERE "user" IN ('{user_list}')
        GROUP BY "user"
    """

    print(f"Sql get user first trade date:\n {full_query}")
    return full_query


def sql_get_user_last_trade_before_date(users: List[str], before_date: datetime) -> str:
    """Get the last trade date before a given date for each user."""
    if not users:
        return ""

    user_list = "', '".join(users)
    before_ts = int(before_date.timestamp())

    start_lookup = before_date - timedelta(days=15)
    days_lookup = (before_date - start_lookup).days
    parts = partition_pred(partition_tuples(start_lookup, days_lookup))

    full_query = f"""
        SELECT 
            "user",
            MAX(CAST(ts AS BIGINT)) as last_trade_ts
        FROM eventtype_orderrecord
        WHERE ({parts}) AND CAST(ts AS BIGINT) < {before_ts}
          AND "user" IN ('{user_list}')
        GROUP BY "user"
    """

    print(f"Sql get user last trade before date:\n {full_query}")
    return full_query


def sql_get_user_volume_since_date(users: List[str], start_date: datetime) -> str:
    """Get trading volume for users since a given date."""
    if not users:
        return ""

    user_list = "', '".join(users)
    start_ts = int(start_date.timestamp())
    now = datetime.now(tz=UTC)
    days_since = (now - start_date).days + 1
    parts = partition_pred(partition_tuples(start_date, days_since))

    full_query = f"""
        WITH user_trades AS (
            SELECT
                taker AS user,
                CAST(quoteassetamountfilled AS DOUBLE) / {QUOTE_ASSET_SCALE_FACTOR} AS quote_volume
            FROM eventtype_traderecord
            WHERE ({parts}) AND CAST(ts AS BIGINT) >= {start_ts}
              AND taker IN ('{user_list}')
            UNION ALL
            SELECT
                maker AS user,
                CAST(quoteassetamountfilled AS DOUBLE) / {QUOTE_ASSET_SCALE_FACTOR} AS quote_volume
            FROM eventtype_traderecord
            WHERE ({parts}) AND CAST(ts AS BIGINT) >= {start_ts}
              AND maker IN ('{user_list}')
        )
        SELECT
            user,
            SUM(quote_volume) AS total_volume
        FROM user_trades
        GROUP BY user
    """

    print(f"Sql get user volume since date:\n {full_query}")
    return full_query


async def calculate_wallet_activity(since_date_str: str) -> Dict[str, Any]:
    conn = None
    try:
        since_date = parser.parse(since_date_str).replace(tzinfo=UTC)
        logger.info(
            f"Connecting to Athena. S3 staging: {S3_OUTPUT}, Region: {REGION}, DB: {DATABASE}"
        )
        conn = connect(
            s3_staging_dir=S3_OUTPUT, region_name=REGION, schema_name=DATABASE
        )
        logger.info("Successfully connected to Athena.")
        logger.info(f"Getting all traders since {since_date_str}...")
        q_all_traders = sql_get_all_traders_since_date(since_date)
        all_traders_df = pd.read_sql(q_all_traders, conn)
        all_traders = all_traders_df["user"].tolist()

        if not all_traders:
            return {
                "analysis_date": since_date_str,
                "new_wallets_count": 0,
                "new_wallets_volume": 0.0,
                "reactivated_wallets_count": 0,
                "reactivated_wallets_volume": 0.0,
                "active_wallets_count": 0,
                "active_wallets_volume": 0.0,
                "total_wallets_count": 0,
                "total_volume": 0.0,
                "wallet_data": [],
            }

        logger.info("Getting first trade dates for all traders...")
        q_first_trades = sql_get_user_first_trade_date(all_traders)
        first_trades_df = pd.read_sql(q_first_trades, conn)
        logger.info("Getting last trade dates before analysis date...")
        q_last_trades = sql_get_user_last_trade_before_date(all_traders, since_date)
        last_trades_df = pd.read_sql(q_last_trades, conn)
        logger.info("Calculating volume since analysis date...")
        q_volume = sql_get_user_volume_since_date(all_traders, since_date)
        volume_df = pd.read_sql(q_volume, conn)

        result = {
            "analysis_date": since_date_str,
            "volume_df": volume_df.to_dict(orient="records"),
            "first_trades_df": first_trades_df.to_dict(orient="records"),
            "last_trades_df": last_trades_df.to_dict(orient="records"),
        }

        logger.info(f"Successfully calculated wallet activity for {since_date_str}.")
        return result

    except Exception as e:
        logger.error(f"Error in calculate_wallet_activity: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to process wallet activity data: {str(e)}"
        )
    finally:
        if conn:
            conn.close()
            logger.info("Athena connection closed.")


@router.get("/calculate", response_model=Dict[str, Any])
async def get_wallet_activity(
    since_date: str = Query(
        ..., description="The date to analyze wallet activity from (YYYY-MM-DD)."
    ),
):
    """
    Analyzes wallet activity since a given date, categorizing wallets into:
    - New Wallets: Wallets that connected for the first time after the given date
    - Reactivated Wallets: Wallets that had no trading activity for more than 15 days prior to the given date, but traded again after
    - Active Wallets: Wallets that did trade within 15 days before the given date, and continued trading after
    """
    try:
        logger.info(f"Received request for /calculate: since_date='{since_date}'")
        result_data = await calculate_wallet_activity(since_date)
        return result_data
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Unhandled error in /calculate endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal server error occurred during calculation.",
        )
