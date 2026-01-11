import argparse
import json
import pandas as pd

from bot.config import load_settings
from bot.exchange.bybit_client import BybitClient
from bot.indicators import prepare_with_indicators
from bot.live import run_live_from_api
from bot.manual import close_position_manual, open_position_manual, show_status
from bot.strategy import build_signals, enrich_for_strategy
from bot.simulation import Simulator


def _timeframe_to_bybit_interval(tf: str) -> str:
    """
    Простейшее сопоставление '15m' -> '15' для unified API Bybit.
    При необходимости можно расширить.
    """
    mapping = {
        "1m": "1",
        "3m": "3",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "4h": "240",
    }
    return mapping.get(tf, "15")


def run_backtest(data_path: str):
    settings = load_settings()
    df_raw = pd.read_csv(data_path)
    df_ind = prepare_with_indicators(
        df_raw,
        adx_length=settings.strategy.adx_length,
        di_length=settings.strategy.di_length,
        sma_length=settings.strategy.sma_length,
        rsi_length=settings.strategy.rsi_length,
        breakout_lookback=settings.strategy.breakout_lookback,
        bb_length=settings.strategy.bb_length,
        bb_std=settings.strategy.bb_std,
    )
    df_ready = enrich_for_strategy(df_ind, settings.strategy)
    signals = build_signals(df_ready, settings.strategy)
    sim = Simulator(settings)
    result = sim.run(df_ready, signals)
    trades_out = [
        {
            "entry_time": str(t.entry_time) if t.entry_time is not None else None,
            "exit_time": str(t.exit_time) if t.exit_time is not None else None,
            "side": t.side.value,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "size_usd": t.size_usd,
            "pnl": t.pnl,
            "entry_reason": t.entry_reason,
            "exit_reason": t.exit_reason,
        }
        for t in result["trades"]
    ]
    print(json.dumps({"total_pnl": result["total_pnl"], "trades": trades_out}, ensure_ascii=False, indent=2))


def run_backtest_from_api(limit: int = 500):
    """
    Берём 15m свечи с Bybit (тестнет/реал в зависимости от настроек),
    считаем индикаторы и гоняем стратегию.
    """
    settings = load_settings()
    client = BybitClient(settings.api)
    interval = _timeframe_to_bybit_interval(settings.timeframe)
    df_raw = client.get_kline_df(symbol=settings.symbol, interval=interval, limit=limit)

    df_ind = prepare_with_indicators(
        df_raw,
        adx_length=settings.strategy.adx_length,
        di_length=settings.strategy.di_length,
        sma_length=settings.strategy.sma_length,
        rsi_length=settings.strategy.rsi_length,
        breakout_lookback=settings.strategy.breakout_lookback,
        bb_length=settings.strategy.bb_length,
        bb_std=settings.strategy.bb_std,
    )
    df_ready = enrich_for_strategy(df_ind, settings.strategy)
    signals = build_signals(df_ready, settings.strategy)
    sim = Simulator(settings)
    result = sim.run(df_ready, signals)
    trades_out = [
        {
            "entry_time": str(t.entry_time) if t.entry_time is not None else None,
            "exit_time": str(t.exit_time) if t.exit_time is not None else None,
            "side": t.side.value,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "size_usd": t.size_usd,
            "pnl": t.pnl,
            "entry_reason": t.entry_reason,
            "exit_reason": t.exit_reason,
        }
        for t in result["trades"]
    ]
    print(json.dumps({"total_pnl": result["total_pnl"], "trades": trades_out}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Bybit futures bot")
    parser.add_argument(
        "--mode",
        choices=["backtest", "backtest_api", "live_api", "manual", "web"],
        default="backtest_api",
        help="Режим работы бота",
    )
    parser.add_argument("--web-port", type=int, default=5000, help="Порт для веб-админки (режим web)")
    parser.add_argument("--web-host", type=str, default="127.0.0.1", help="Хост для веб-админки (режим web)")
    parser.add_argument("--data", help="CSV with 15m candles")
    parser.add_argument("--limit", type=int, default=500, help="Количество свечей для backtest_api")
    
    # Параметры для ручного режима
    parser.add_argument("--action", choices=["open", "close", "status"], help="Действие в режиме manual")
    parser.add_argument("--side", choices=["long", "short"], help="Сторона позиции (для open)")
    parser.add_argument("--qty", type=float, help="Количество монет (для open)")
    parser.add_argument("--usd", type=float, help="Размер позиции в USD (для open)")
    
    args = parser.parse_args()

    if args.mode == "backtest":
        if not args.data:
            raise SystemExit("--data обязателен в режиме backtest")
        run_backtest(args.data)
    elif args.mode == "backtest_api":
        run_backtest_from_api(limit=args.limit)
    elif args.mode == "live_api":
        settings = load_settings()
        run_live_from_api(settings)
    elif args.mode == "manual":
        settings = load_settings()
        if args.action == "status":
            show_status(settings)
        elif args.action == "open":
            if not args.side:
                raise SystemExit("--side обязателен для --action open (long или short)")
            open_position_manual(settings, side=args.side, qty=args.qty, usd_amount=args.usd)
        elif args.action == "close":
            close_position_manual(settings)
        else:
            raise SystemExit("--action обязателен в режиме manual (status, open, close)")
    elif args.mode == "web":
        from bot.web.app import run_web_server
        print(f"[web] Starting admin panel on http://{args.web_host}:{args.web_port}")
        run_web_server(host=args.web_host, port=args.web_port, debug=False)


if __name__ == "__main__":
    main()

