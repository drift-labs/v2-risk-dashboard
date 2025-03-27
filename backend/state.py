import os
from asyncio import create_task, gather
from datetime import datetime

from anchorpy.provider import Wallet
from driftpy.account_subscription_config import AccountSubscriptionConfig
from driftpy.drift_client import DriftClient
from driftpy.market_map.market_map import MarketMap
from driftpy.market_map.market_map_config import MarketMapConfig
from driftpy.market_map.market_map_config import (
    WebsocketConfig as MarketMapWebsocketConfig,
)
from driftpy.pickle.vat import Vat
from driftpy.types import MarketType
from driftpy.user_map.user_map import UserMap
from driftpy.user_map.user_map_config import UserMapConfig, UserStatsMapConfig
from driftpy.user_map.user_map_config import (
    WebsocketConfig as UserMapWebsocketConfig,
)
from driftpy.user_map.userstats_map import UserStatsMap
from fastapi import Request
from solana.rpc.async_api import AsyncClient

from backend.utils.vat import load_newest_files
from backend.utils.waiting_for import waiting_for


class BackendState:
    connection: AsyncClient
    dc: DriftClient
    spot_map: MarketMap
    perp_map: MarketMap
    user_map: UserMap
    stats_map: UserStatsMap

    current_pickle_path: str
    last_oracle_slot: int
    vat: Vat
    ready: bool
    
    def __init__(self):
        """Initialize critical attributes with default values to prevent access errors"""
        self.ready = False
        self.current_pickle_path = "bootstrap"
        self.last_oracle_slot = 0

    def initialize(
        self, url: str
    ):  # Not using __init__ because we need the rpc url to be passed in
        self.connection = AsyncClient(url)
        self.dc = DriftClient(
            self.connection,
            Wallet.dummy(),
            "mainnet",
            account_subscription=AccountSubscriptionConfig("cached"),
        )
        self.perp_map = MarketMap(
            MarketMapConfig(
                self.dc.program,
                MarketType.Perp(),
                MarketMapWebsocketConfig(),
                self.dc.connection,
            )
        )
        self.spot_map = MarketMap(
            MarketMapConfig(
                self.dc.program,
                MarketType.Spot(),
                MarketMapWebsocketConfig(),
                self.dc.connection,
            )
        )
        self.user_map = UserMap(UserMapConfig(self.dc, UserMapWebsocketConfig()))
        self.stats_map = UserStatsMap(UserStatsMapConfig(self.dc))
        self.vat = Vat(
            self.dc,
            self.user_map,
            self.stats_map,
            self.spot_map,
            self.perp_map,
        )
        self.ready = False
        self.current_pickle_path = "bootstrap"

    async def bootstrap(self):
        with waiting_for("drift client"):
            await self.dc.subscribe()
        with waiting_for("subscriptions"):
            await gather(
                create_task(self.spot_map.subscribe()),
                create_task(self.perp_map.subscribe()),
                create_task(self.user_map.subscribe()),
                create_task(self.stats_map.subscribe()),
            )
        self.current_pickle_path = "bootstrap"

    async def take_pickle_snapshot(self):
        now = datetime.now()
        folder_name = now.strftime("vat-%Y-%m-%d-%H-%M-%S")
        if not os.path.exists("pickles"):
            os.makedirs("pickles")
        path = os.path.join("pickles", folder_name, "")

        os.makedirs(path, exist_ok=True)
        with waiting_for("pickling"):
            result = await self.vat.pickle(path)
        with waiting_for("unpickling"):
            await self.load_pickle_snapshot(path)
        return result

    async def load_pickle_snapshot(self, directory: str):
        pickle_map = load_newest_files(directory)
        self.current_pickle_path = os.path.realpath(directory)
        with waiting_for("unpickling"):
            await self.vat.unpickle(
                users_filename=pickle_map["usermap"],
                user_stats_filename=pickle_map["userstats"],
                spot_markets_filename=pickle_map["spot"],
                perp_markets_filename=pickle_map["perp"],
                spot_oracles_filename=pickle_map["spotoracles"],
                perp_oracles_filename=pickle_map["perporacles"],
            )

        self.last_oracle_slot = int(
            pickle_map["perporacles"].split("_")[-1].split(".")[0]
        )
        return pickle_map

    async def close(self):
        await self.dc.unsubscribe()
        await self.connection.close()


class BackendRequest(Request):
    @property
    def backend_state(self) -> BackendState:
        return self.state.get("backend_state")

    @backend_state.setter
    def backend_state(self, value: BackendState):
        self.state["backend_state"] = value
