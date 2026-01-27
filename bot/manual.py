"""
Ручное управление позициями через CLI.
"""
import time
from typing import Optional

from bot.config import AppSettings
from bot.exchange.bybit_client import BybitClient
from bot.live import _calculate_order_qty, _get_balance, _get_position, _get_open_orders


def get_current_price(client: BybitClient, symbol: str) -> Optional[float]:
    """Получить текущую цену символа."""
    try:
        df = client.get_kline_df(symbol=symbol, interval="1", limit=1)
        if not df.empty:
            return float(df["close"].iloc[-1])
        return None
    except Exception as e:
        print(f"Error getting current price: {e}")
        return None


def open_position_manual(
    settings: AppSettings,
    side: str,  # "long" or "short"
    qty: Optional[float] = None,
    usd_amount: Optional[float] = None,
):
    """
    Ручное открытие позиции.
    
    Args:
        side: "long" или "short"
        qty: количество монет (если указано, используется напрямую)
        usd_amount: размер позиции в USD (если указано, рассчитывается qty)
    """
    client = BybitClient(settings.api)
    
    # Проверяем наличие API ключей
    if not settings.api.api_key or not settings.api.api_secret:
        print("❌ ERROR: API keys are missing!")
        return
    
    # Получаем текущую цену
    price = get_current_price(client, settings.symbol)
    if price is None:
        print("❌ Could not get current price")
        return
    
    # Получаем баланс
    balance = _get_balance(client)
    if balance is None:
        print("⚠️  Could not get balance")
    
    # Проверяем существующую позицию
    position = _get_position(client, settings.symbol)
    if position:
        print(f"⚠️  Warning: Already have {position['side']} position: {position['size']:.3f} @ {position['avg_price']:.2f}")
        response = input("Close existing position first? (y/n): ")
        if response.lower() == 'y':
            close_position_manual(settings)
            time.sleep(1)
        else:
            print("Aborted")
            return
    
    # Проверяем открытые ордера
    orders = _get_open_orders(client, settings.symbol)
    if orders:
        print(f"⚠️  Warning: {len(orders)} open orders exist")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted")
            return
    
    # Рассчитываем qty
    if qty is None:
        if usd_amount is None:
            # Используем настройки по умолчанию
            usd_amount = settings.risk.base_order_usd
        qty = _calculate_order_qty(
            desired_usd=usd_amount,
            balance=balance,
            price=price,
            leverage=settings.leverage,
            balance_percent=settings.risk.balance_percent_per_trade,
        )
    
    if qty <= 0:
        print(f"❌ Calculated qty too small: {qty}")
        return
    
    # Определяем side для ордера
    order_side = "Buy" if side.lower() == "long" else "Sell"
    
    # Рассчитываем TP и SL
    if side.lower() == "long":
        stop_loss_price = price * (1 - settings.risk.stop_loss_pct)
        take_profit_price = price * (1 + settings.risk.take_profit_pct)
    else:
        stop_loss_price = price * (1 + settings.risk.stop_loss_pct)
        take_profit_price = price * (1 - settings.risk.take_profit_pct)
    
    position_value = qty * price
    margin_required = position_value / settings.leverage
    
    print(f"\n{'='*60}")
    print(f"Opening {side.upper()} position:")
    print(f"  Symbol: {settings.symbol}")
    print(f"  Quantity: {qty:.3f}")
    print(f"  Entry price: ${price:.2f}")
    print(f"  Position value: ${position_value:.2f}")
    print(f"  Margin required: ${margin_required:.2f}")
    print(f"  Leverage: {settings.leverage}x")
    print(f"  Stop Loss: ${stop_loss_price:.2f} ({settings.risk.stop_loss_pct*100:.1f}%)")
    print(f"  Take Profit: ${take_profit_price:.2f} ({settings.risk.take_profit_pct*100:.1f}%)")
    print(f"{'='*60}")
    
    response = input("\nConfirm order? (y/n): ")
    if response.lower() != 'y':
        print("Aborted")
        return
    
    # Открываем позицию
    try:
        resp = client.place_order(symbol=settings.symbol, side=order_side, qty=qty)
        if resp.get("retCode") == 0:
            print(f"✅ Order placed: {resp.get('retMsg', 'OK')}")
            # Устанавливаем TP и SL
            time.sleep(0.5)
            tp_sl_resp = client.set_trading_stop(
                symbol=settings.symbol,
                stop_loss=stop_loss_price,
                take_profit=take_profit_price,
            )
            if tp_sl_resp.get("retCode") == 0:
                print(f"✅ TP and SL set successfully")
            else:
                print(f"⚠️  Failed to set TP/SL: {tp_sl_resp.get('retMsg', 'Unknown error')}")
        else:
            print(f"❌ Order failed: {resp.get('retMsg', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Error: {e}")


def close_position_manual(settings: AppSettings):
    """Ручное закрытие позиции."""
    client = BybitClient(settings.api)
    
    # Проверяем наличие API ключей
    if not settings.api.api_key or not settings.api.api_secret:
        print("❌ ERROR: API keys are missing!")
        return
    
    # Получаем текущую позицию
    position = _get_position(client, settings.symbol)
    if not position:
        print("ℹ️  No open position")
        return
    
    print(f"\n{'='*60}")
    print(f"Current position:")
    print(f"  Side: {position['side']}")
    print(f"  Size: {position['size']:.3f}")
    print(f"  Avg Price: ${position['avg_price']:.2f}")
    print(f"  Mark Price: ${position['mark_price']:.2f}")
    print(f"  Unrealised PnL: ${position['unrealised_pnl']:.2f}")
    print(f"{'='*60}")
    
    response = input("\nClose position? (y/n): ")
    if response.lower() != 'y':
        print("Aborted")
        return
    
    # Закрываем позицию
    side = "Sell" if position["side"] == "Buy" else "Buy"
    qty = position["size"]
    
    try:
        resp = client.place_order(
            symbol=settings.symbol,
            side=side,
            qty=qty,
            reduce_only=True,
        )
        if resp.get("retCode") == 0:
            print(f"✅ Position closed: {resp.get('retMsg', 'OK')}")
        else:
            print(f"❌ Failed to close: {resp.get('retMsg', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Error: {e}")


def show_status(settings: AppSettings):
    """Показать текущий статус: баланс, позиция, ордера."""
    client = BybitClient(settings.api)
    
    balance = _get_balance(client)
    position = _get_position(client, settings.symbol)
    orders = _get_open_orders(client, settings.symbol)
    price = get_current_price(client, settings.symbol)
    
    print(f"\n{'='*60}")
    print(f"Status for {settings.symbol}:")
    print(f"{'='*60}")
    print(f"Current Price: ${price:.2f}" if price else "Current Price: N/A")
    print(f"Balance: {balance:.2f} USDT" if balance is not None else "Balance: N/A")
    
    if position:
        print(f"\nPosition:")
        print(f"  Side: {position['side']}")
        print(f"  Size: {position['size']:.3f}")
        print(f"  Avg Price: ${position['avg_price']:.2f}")
        print(f"  Mark Price: ${position['mark_price']:.2f}")
        print(f"  Unrealised PnL: ${position['unrealised_pnl']:.2f}")
        print(f"  Leverage: {position['leverage']}x")
    else:
        print("\nPosition: None")
    
    if orders:
        print(f"\nOpen Orders ({len(orders)}):")
        for order in orders[:5]:  # показываем первые 5
            price_str = f"${order['price']:.2f}" if order['price'] else 'Market'
            print(f"  - {order['side']} {order['qty']:.3f} @ {price_str}")
    else:
        print("\nOpen Orders: None")
    
    print(f"{'='*60}\n")




