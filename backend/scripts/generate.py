"""
Pre-generate results for CPU-heavy endpoints.

Replaces gen.sh, run_gen_loop.sh, and generate_ucache.py with a single script.

Usage:
    # One-shot generation (takes fresh snapshot first)
    python -m backend.scripts.generate

    # One-shot using existing pickle
    python -m backend.scripts.generate --use-snapshot

    # Continuous generation (for production, replaces run_gen_loop.sh)
    python -m backend.scripts.generate --loop
"""

import argparse
import asyncio
import glob
import json
import logging
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

from dotenv import load_dotenv

from backend.results import RESULTS_DIR, make_result_key

load_dotenv()
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Job definitions
# ---------------------------------------------------------------------------


@dataclass
class Job:
    """A single heavy computation to pre-generate."""

    endpoint: str
    params: dict = field(default_factory=dict)

    @property
    def result_key(self) -> str:
        return make_result_key(self.endpoint, self.params)


def get_jobs() -> list[Job]:
    """All heavy endpoint parameter combinations to pre-generate."""
    jobs = []

    # Asset liability
    jobs.append(Job("asset-liability/matrix", {"mode": 0, "perp_market_index": 0}))

    # Price shock: 2 asset groups x 2 (distortion, scenario) pairs x 4 pool filters
    for asset_group in ["ignore stables", "jlp only"]:
        for distortion, scenarios in [(0.05, 5), (0.1, 10)]:
            for pool_id in [None, 0, 1, 3]:
                params = {
                    "asset_group": asset_group,
                    "oracle_distortion": distortion,
                    "n_scenarios": scenarios,
                }
                if pool_id is not None:
                    params["pool_id"] = pool_id
                jobs.append(Job("price-shock/usermap", params))

    return jobs


# ---------------------------------------------------------------------------
# Worker function (runs in separate process via ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def run_job(pickle_path: str, job_endpoint: str, job_params: dict, job_key: str) -> str:
    """Execute a single job in its own process. Returns status message.

    Note: We pass primitives instead of a Job dataclass to avoid pickling issues
    across process boundaries.
    """
    return asyncio.run(_run_job_async(pickle_path, job_endpoint, job_params, job_key))


async def _run_job_async(
    pickle_path: str, endpoint: str, params: dict, result_key: str
) -> str:
    """Async implementation of job execution."""
    from backend.api.asset_liability import _get_asset_liability_matrix
    from backend.api.price_shock import _get_price_shock
    from backend.state import BackendState

    state = BackendState()
    state.initialize(os.getenv("RPC_URL", ""))
    await state.load_pickle_snapshot(pickle_path)

    try:
        if endpoint == "price-shock/usermap":
            content = await _get_price_shock(
                state.last_oracle_slot,
                state.vat,
                state.dc,
                oracle_distortion=params["oracle_distortion"],
                asset_group=params["asset_group"],
                n_scenarios=params["n_scenarios"],
                pool_id=params.get("pool_id"),
            )
        elif endpoint == "asset-liability/matrix":
            content = await _get_asset_liability_matrix(
                state.last_oracle_slot,
                state.vat,
                mode=params["mode"],
                perp_market_index=params["perp_market_index"],
            )
        else:
            return f"Unknown endpoint: {endpoint}"

        result_file = os.path.join(RESULTS_DIR, f"{result_key}.json")
        tmp_file = result_file + ".tmp"
        with open(tmp_file, "w") as f:
            json.dump(content, f)
        os.replace(tmp_file, result_file)

        return f"OK: {result_key}"
    finally:
        await state.close()


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------


async def take_fresh_snapshot():
    """Bootstrap from RPC and take a fresh pickle snapshot."""
    from backend.state import BackendState

    state = BackendState()
    state.initialize(os.getenv("RPC_URL", ""))
    logger.info("Bootstrapping from RPC...")
    await state.bootstrap()
    await state.take_pickle_snapshot()
    await state.close()
    logger.info("Fresh pickle snapshot taken.")


def get_latest_pickle() -> str:
    paths = sorted(glob.glob("pickles/*"))
    if not paths:
        raise RuntimeError("No pickle snapshots found in pickles/")
    return paths[-1]


def cleanup_old_pickles(keep: int = 3):
    """Remove old pickle directories, keeping the newest N."""
    pickle_dirs = sorted(glob.glob("pickles/*"), key=os.path.getmtime, reverse=True)
    for old_dir in pickle_dirs[keep:]:
        shutil.rmtree(old_dir, ignore_errors=True)
        logger.info(f"Removed old pickle: {old_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate_all(use_snapshot: bool = False):
    """Run all jobs. If not use_snapshot, takes a fresh pickle first."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not use_snapshot:
        logger.info("Taking fresh pickle snapshot...")
        asyncio.run(take_fresh_snapshot())

    pickle_path = get_latest_pickle()
    jobs = get_jobs()

    logger.info(f"Generating {len(jobs)} results using pickle: {pickle_path}")

    max_workers = min(8, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                run_job, pickle_path, job.endpoint, job.params, job.result_key
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                logger.info(result)
            except Exception as e:
                logger.error(f"FAILED: {job.result_key}: {e}")

    cleanup_old_pickles()
    logger.info("Generation complete.")


def main():
    parser = argparse.ArgumentParser(description="Generate pre-computed results")
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously (replaces run_gen_loop.sh)",
    )
    parser.add_argument(
        "--use-snapshot",
        action="store_true",
        help="Use existing pickle instead of taking a fresh one",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=3600,
        help="Seconds between loops (default: 3600)",
    )
    parser.add_argument(
        "--retry-interval",
        type=int,
        default=300,
        help="Seconds to wait after failure (default: 300)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.loop:
        while True:
            try:
                generate_all(use_snapshot=args.use_snapshot)
                logger.info(f"Sleeping {args.interval}s until next generation...")
                time.sleep(args.interval)
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                import traceback

                traceback.print_exc()
                logger.info(f"Retrying in {args.retry_interval}s...")
                time.sleep(args.retry_interval)
    else:
        generate_all(use_snapshot=args.use_snapshot)


if __name__ == "__main__":
    main()
