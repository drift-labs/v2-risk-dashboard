import argparse
import asyncio
import glob
import json
import os
from dataclasses import dataclass
from itertools import islice

from dotenv import load_dotenv
from fastapi.responses import JSONResponse

from backend.api.asset_liability import _get_asset_liability_matrix
from backend.api.price_shock import _get_price_shock
from backend.state import BackendState
from shared.types import PriceShockAssetGroup

load_dotenv()


def chunk_list(lst, n):
    """Split list into n chunks"""
    size = len(lst)
    chunk_size = (size + n - 1) // n
    it = iter(lst)
    return [list(islice(it, chunk_size)) for _ in range(n)]


@dataclass
class Endpoint:
    endpoint: str
    params: dict


@dataclass
class AssetLiabilityEndpoint(Endpoint):
    mode: int
    perp_market_index: int

    def __init__(self, mode: int, perp_market_index: int):
        self.mode = mode
        self.perp_market_index = perp_market_index
        super().__init__(
            "asset-liability/matrix",
            {"mode": mode, "perp_market_index": perp_market_index},
        )


@dataclass
class PriceShockEndpoint(Endpoint):
    asset_group: PriceShockAssetGroup
    oracle_distortion: float
    n_scenarios: int
    pool_id: int

    def __init__(
        self,
        asset_group: PriceShockAssetGroup,
        oracle_distortion: float,
        n_scenarios: int,
        pool_id: int = None,
    ):
        self.asset_group = asset_group
        self.oracle_distortion = oracle_distortion
        self.n_scenarios = n_scenarios
        self.pool_id = pool_id
        
        params = {
            "asset_group": asset_group,
            "oracle_distortion": oracle_distortion,
            "n_scenarios": n_scenarios,
        }
        
        # Only add pool_id if it's not None
        if pool_id is not None:
            params["pool_id"] = pool_id
            
        super().__init__(
            "price-shock/usermap",
            params,
        )


async def process_multiple_endpoints(state_pickle_path: str, endpoints: list[Endpoint]):
    """Process a single endpoint in its own process"""
    state = BackendState()
    state.initialize(os.getenv("RPC_URL") or "")
    await state.load_pickle_snapshot(state_pickle_path)

    results = []

    for endpoint_object in endpoints:
        endpoint = endpoint_object.endpoint
        query_params = endpoint_object.params

        class MockRequest:
            def __init__(self, path: str, query_params: dict):
                self.method = "GET"
                query_string = "&".join([f"{k}={v}" for k, v in query_params.items()])
                self.url = type("URL", (), {"path": path, "query": query_string})()

        request = MockRequest(f"/api/{endpoint}", query_params)

        async def mock_call_next(request):
            if endpoint == "price-shock/usermap":
                content = await _get_price_shock(
                    state.last_oracle_slot,
                    state.vat,
                    state.dc,
                    oracle_distortion=query_params["oracle_distortion"],
                    asset_group=query_params["asset_group"],
                    n_scenarios=query_params["n_scenarios"],
                    pool_id=query_params.get("pool_id"),
                )

            if endpoint == "asset-liability/matrix":
                content = await _get_asset_liability_matrix(
                    state.last_oracle_slot,
                    state.vat,
                    mode=query_params["mode"],
                    perp_market_index=query_params["perp_market_index"],
                )

            return JSONResponse(content=content)

        print(request.url.path)
        print(request.url.query)

        ucache_key = f"{request.method}{request.url.path}"
        if request.url.query:
            safe_query = request.url.query.replace("&", "_").replace("=", "-")
            ucache_key = f"{ucache_key}__{safe_query}"
        ucache_key = ucache_key.replace("/", "_")
        ucache_file = os.path.join("ucache", f"{ucache_key}.json")

        async def run_request():
            response = await mock_call_next(request)
            if response.status_code == 200:
                response_data = {
                    "content": json.loads(response.body.decode()),
                    "status_code": response.status_code,
                    "headers": {"content-type": "application/json"},
                }

                with open(ucache_file, "w") as f:
                    json.dump(response_data, f, indent=2)
                return f"Generated cache for {endpoint}"

        await run_request()

    await state.close()
    return results


async def generate_ucache(endpoints: list[Endpoint]):
    """Generate ucache files by splitting endpoints across processes"""
    ucache_dir = "ucache"

    print(f"Generating ucache for endpoints: {endpoints}")
    if not os.path.exists(ucache_dir):
        os.makedirs(ucache_dir)
    state_pickle_path = sorted(glob.glob("pickles/*"))[-1]
    await process_multiple_endpoints(state_pickle_path, endpoints)


async def main():
    parser = argparse.ArgumentParser(description="Generate ucache files")
    parser.add_argument(
        "--use-snapshot", action="store_true", help="Use existing snapshot"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    al_parser = subparsers.add_parser("asset-liability")
    al_parser.add_argument("--mode", type=int, required=True)
    al_parser.add_argument("--perp-market-index", type=int, required=True)

    ps_parser = subparsers.add_parser("price-shock")
    ps_parser.add_argument("--asset-group", type=str, required=True)
    ps_parser.add_argument("--oracle-distortion", type=float, required=True)
    ps_parser.add_argument("--n-scenarios", type=int, required=True)
    ps_parser.add_argument("--pool-id", type=int, help="Filter by pool ID (optional)")

    args = parser.parse_args()

    if not args.use_snapshot:
        state = BackendState()
        state.initialize(os.getenv("RPC_URL") or "")
        print("Taking snapshot")
        await state.bootstrap()
        await state.take_pickle_snapshot()
        await state.close()

    endpoints = []
    if args.command == "asset-liability":
        endpoints.append(
            AssetLiabilityEndpoint(
                mode=args.mode, perp_market_index=args.perp_market_index
            )
        )
    elif args.command == "price-shock":
        endpoints.append(
            PriceShockEndpoint(
                asset_group=args.asset_group,
                oracle_distortion=args.oracle_distortion,
                n_scenarios=args.n_scenarios,
                pool_id=getattr(args, 'pool_id', None),
            )
        )

    await generate_ucache(endpoints)


if __name__ == "__main__":
    asyncio.run(main())

# Usage example:
# python -m backend.scripts.generate_ucache --use-snapshot asset-liability --mode 0 --perp-market-index 0
# python -m backend.scripts.generate_ucache --use-snapshot price-shock --asset-group "ignore+stables" --oracle-distortion 0.05 --n-scenarios 5
# python -m backend.scripts.generate_ucache --use-snapshot price-shock --asset-group "ignore+stables" --oracle-distortion 0.05 --n-scenarios 5 --pool-id 0
