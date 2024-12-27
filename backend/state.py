import os
from asyncio import create_task, gather
from datetime import datetime
import logging
from typing import Optional

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

logger = logging.getLogger(__name__)


class BackendState:
    connection: AsyncClient
    dc: DriftClient
    spot_markets: MarketMap
    perp_markets: MarketMap
    user_map: UserMap
    stats_map: UserStatsMap

    current_pickle_path: str
    last_oracle_slot: int
    vat: Vat
    ready: bool

    def __init__(self):
        self.ready = False
        self.vat = None
        self.dc = None
        self.connection = None
        self.spot_markets = None
        self.perp_markets = None
        self.user_map = None
        self.stats_map = None
        self.current_pickle_path = None
        self.last_oracle_slot = 0
        logger.info("BackendState initialized")

    def initialize(self, url: str):
        """Initialize the backend state with RPC URL."""
        logger.info(f"Initializing backend state with URL: {url}")
        try:
            self.connection = AsyncClient(url)
            self.dc = DriftClient(
                self.connection,
                Wallet.dummy(),
                "mainnet",
                account_subscription=AccountSubscriptionConfig("cached"),
            )
            logger.info("Created DriftClient")
            
            self.perp_markets = MarketMap(
                MarketMapConfig(
                    self.dc.program,
                    MarketType.Perp(),
                    MarketMapWebsocketConfig(),
                    self.dc.connection,
                )
            )
            logger.info("Created perp_markets")
            
            self.spot_markets = MarketMap(
                MarketMapConfig(
                    self.dc.program,
                    MarketType.Spot(),
                    MarketMapWebsocketConfig(),
                    self.dc.connection,
                )
            )
            logger.info("Created spot_markets")
            
            self.user_map = UserMap(UserMapConfig(self.dc, UserMapWebsocketConfig()))
            logger.info("Created user_map")
            
            self.stats_map = UserStatsMap(UserStatsMapConfig(self.dc))
            logger.info("Created stats_map")
            
            # Don't create VAT here, it will be created after subscriptions in bootstrap
            logger.info("Backend state initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize backend state: {e}")
            self.ready = False
            raise

    async def bootstrap(self):
        """Bootstrap the backend state."""
        logger.info("Starting bootstrap process")
        try:
            with waiting_for("drift client"):
                await self.dc.subscribe()
            with waiting_for("subscriptions"):
                await gather(
                    create_task(self.spot_markets.subscribe()),
                    create_task(self.perp_markets.subscribe()),
                    create_task(self.user_map.subscribe()),
                    create_task(self.stats_map.subscribe()),
                )
            
            logger.info("Creating VAT object...")
            self.vat = Vat(
                self.dc,
                self.user_map,
                self.stats_map,
                self.spot_markets,
                self.perp_markets,
            )
            logger.info("VAT object created successfully")
            logger.info(f"VAT attributes: {dir(self.vat)}")
            
            # Set ready flag only after successful bootstrap
            self.ready = True
            logger.info("Bootstrap completed successfully")
        except Exception as e:
            logger.error(f"Bootstrap failed: {e}")
            self.ready = False
            raise

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
        """Load state from pickle snapshot."""
        logger.info(f"Loading pickle snapshot from {directory}")
        try:
            pickle_map = load_newest_files(directory)
            self.current_pickle_path = os.path.realpath(directory)
            
            logger.info("Starting VAT unpickling process...")
            logger.info(f"Pickle map: {pickle_map}")
            
            # Create VAT object before unpickling
            if self.vat is None:
                logger.info("Creating new VAT object...")
                self.vat = Vat(
                    self.dc,
                    self.user_map,
                    self.stats_map,
                    self.spot_markets,
                    self.perp_markets,
                )
                logger.info("VAT object created successfully")
                logger.info(f"Initial VAT attributes: {dir(self.vat)}")
                logger.info(f"Initial VAT type: {type(self.vat)}")
            
            with waiting_for("unpickling"):
                logger.info("Unpickling VAT with files:")
                logger.info(f"- Users: {pickle_map['usermap']}")
                logger.info(f"- User Stats: {pickle_map['userstats']}")
                logger.info(f"- Spot Markets: {pickle_map['spot']}")
                logger.info(f"- Perp Markets: {pickle_map['perp']}")
                logger.info(f"- Spot Oracles: {pickle_map['spotoracles']}")
                logger.info(f"- Perp Oracles: {pickle_map['perporacles']}")
                
                await self.vat.unpickle(
                    users_filename=pickle_map["usermap"],
                    user_stats_filename=pickle_map["userstats"],
                    spot_markets_filename=pickle_map["spot"],
                    perp_markets_filename=pickle_map["perp"],
                    spot_oracles_filename=pickle_map["spotoracles"],
                    perp_oracles_filename=pickle_map["perporacles"],
                )
                
                logger.info("VAT unpickling complete")
                logger.info(f"VAT attributes after unpickling: {dir(self.vat)}")
                logger.info(f"VAT type after unpickling: {type(self.vat)}")
                logger.info(f"VAT spot markets available: {hasattr(self.vat, 'spot_markets')}")
                logger.info(f"VAT perp markets available: {hasattr(self.vat, 'perp_markets')}")
                if hasattr(self.vat, 'spot_markets'):
                    logger.info(f"Spot markets type: {type(self.vat.spot_markets)}")
                    logger.info(f"Spot markets attributes: {dir(self.vat.spot_markets)}")
                if hasattr(self.vat, 'perp_markets'):
                    logger.info(f"Perp markets type: {type(self.vat.perp_markets)}")
                    logger.info(f"Perp markets attributes: {dir(self.vat.perp_markets)}")

            self.last_oracle_slot = int(
                pickle_map["perporacles"].split("_")[-1].split(".")[0]
            )
            
            # Set ready flag only after successful load
            self.ready = True
            logger.info("Pickle snapshot loaded successfully")
            return pickle_map
            
        except Exception as e:
            logger.error(f"Failed to load pickle snapshot: {e}")
            self.ready = False
            raise

    async def close(self):
        await self.dc.unsubscribe()
        await self.connection.close()

    @property
    def is_ready(self) -> bool:
        """
        Check if the backend state is ready.
        """
        try:
            logger.debug("Checking readiness")
            
            # Check if state is initialized
            if not self.ready:
                logger.debug("State not initialized")
                return False
            
            # Check if VAT is initialized
            if not self.vat:
                logger.debug("VAT not initialized")
                return False
            
            # Check if DC is initialized
            if not self.dc:
                logger.debug("DC not initialized")
                return False
            
            # Check if connection is initialized
            if not self.connection:
                logger.debug("Connection not initialized")
                return False
            
            # Get market maps
            spot_market_map = None
            perp_market_map = None
            
            if hasattr(self.vat, "spot_markets"):
                if hasattr(self.vat.spot_markets, "market_map"):
                    spot_market_map = self.vat.spot_markets.market_map
                elif hasattr(self.vat.spot_markets, "markets"):
                    spot_market_map = self.vat.spot_markets.markets
            
            if hasattr(self.vat, "perp_markets"):
                if hasattr(self.vat.perp_markets, "market_map"):
                    perp_market_map = self.vat.perp_markets.market_map
                elif hasattr(self.vat.perp_markets, "markets"):
                    perp_market_map = self.vat.perp_markets.markets
            
            if not spot_market_map or not perp_market_map:
                logger.debug("Market maps not available")
                return False
            
            # Check if market maps have data
            try:
                spot_markets_ready = False
                perp_markets_ready = False
                
                # Check spot markets
                if spot_market_map:
                    for _ in spot_market_map:
                        spot_markets_ready = True
                        break
                
                # Check perp markets
                if perp_market_map:
                    for _ in perp_market_map:
                        perp_markets_ready = True
                        break
                
                if not spot_markets_ready or not perp_markets_ready:
                    logger.debug("Market maps empty")
                    return False
                
            except Exception as e:
                logger.error(f"Error checking market maps: {e}")
                return False
            
            logger.debug("All components ready")
            return True
            
        except Exception as e:
            logger.error(f"Error checking readiness: {e}")
            return False


class BackendRequest(Request):
    @property
    def backend_state(self) -> BackendState:
        return self.state.get("backend_state")

    @backend_state.setter
    def backend_state(self, value: BackendState):
        self.state["backend_state"] = value
