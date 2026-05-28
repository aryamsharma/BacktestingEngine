from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Engine.Core.Strategy import Strategy


class Mode(Enum):
    BACKTEST = "backtest"
    PAPER = "paper"


def create(mode: Mode, strategy: Strategy, config: dict):
    if mode == Mode.BACKTEST:
        from Engine.Backtester.Broker import Broker
        from Engine.Backtester.Exchange import Exchange

        exchange = Exchange(
            filepath=config["data_path"],
            sorting=config.get("sorting", False),
            limit=config.get("limit", -1),
            slippage=config.get("slippage", 1),
            lazy_loading=config.get("lazy_loading", None),
            lazy_loading_limit=config.get("lazy_loading_limit", 24),
        )
        return Broker(
            cash=config["cash"],
            algorithm=strategy,
            trade_cost=config.get("trade_cost", 0),
            exchange_feed=exchange,
        )

    if mode == Mode.PAPER:
        from Engine.Papertrader.PaperTradingEngine import PaperTradingEngine

        return PaperTradingEngine(
            strategy=strategy,
            host=config.get("host", "127.0.0.1"),
            port=config.get("port", 7497),
            client_id=config.get("client_id", 1),
            symbols=config["symbols"],
            cash=config.get("cash", 0),
        )

    raise ValueError(f"Unknown mode: {mode}")
