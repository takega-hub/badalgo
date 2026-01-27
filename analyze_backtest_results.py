"""Анализ результатов бэктеста TREND стратегии."""
import pandas as pd

# Загружаем данные
df = pd.read_csv('backtest_trend_BTCUSDT_mtf.csv')

print("=" * 80)
print("📊 АНАЛИЗ РЕЗУЛЬТАТОВ БЭКТЕСТА TREND СТРАТЕГИИ")
print("=" * 80)

print("\n📈 Распределение причин выхода:")
print(df['exit_reason'].value_counts())

print("\n💰 PnL по причинам выхода:")
exit_pnl = df.groupby('exit_reason')['pnl'].agg(['count', 'sum', 'mean']).round(2)
print(exit_pnl)

print("\n📊 Статистика по типам сигналов:")
pullback_trades = df[df['entry_reason'].str.contains('pullback', case=False, na=False)]
breakout_trades = df[df['entry_reason'].str.contains('breakout', case=False, na=False)]

print(f"\nPullback сигналы:")
print(f"  Всего сделок: {len(pullback_trades)}")
print(f"  Прибыльных: {len(pullback_trades[pullback_trades['pnl'] > 0])}")
print(f"  Убыточных: {len(pullback_trades[pullback_trades['pnl'] < 0])}")
print(f"  Win Rate: {len(pullback_trades[pullback_trades['pnl'] > 0]) / len(pullback_trades) * 100:.2f}%")
print(f"  Общий PnL: ${pullback_trades['pnl'].sum():.2f}")

print(f"\nBreakout сигналы:")
print(f"  Всего сделок: {len(breakout_trades)}")
print(f"  Прибыльных: {len(breakout_trades[breakout_trades['pnl'] > 0])}")
print(f"  Убыточных: {len(breakout_trades[breakout_trades['pnl'] < 0])}")
if len(breakout_trades) > 0:
    print(f"  Win Rate: {len(breakout_trades[breakout_trades['pnl'] > 0]) / len(breakout_trades) * 100:.2f}%")
    print(f"  Общий PnL: ${breakout_trades['pnl'].sum():.2f}")

print("\n🔍 Анализ соотношения Win/Loss:")
wins = df[df['pnl'] > 0]
losses = df[df['pnl'] < 0]
if len(wins) > 0 and len(losses) > 0:
    avg_win = wins['pnl'].mean()
    avg_loss = abs(losses['pnl'].mean())
    print(f"  Средний выигрыш: ${avg_win:.2f}")
    print(f"  Средний проигрыш: ${avg_loss:.2f}")
    print(f"  Соотношение: {avg_win / avg_loss:.2f}:1")
    print(f"  Необходимый Win Rate для безубыточности: {1 / (1 + avg_win/avg_loss) * 100:.1f}%")
    print(f"  Текущий Win Rate: {len(wins) / len(df) * 100:.1f}%")

print("\n⏱️ Анализ длительности сделок:")
df['entry_time'] = pd.to_datetime(df['entry_time'])
df['exit_time'] = pd.to_datetime(df['exit_time'])
df['duration'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 3600
print(f"  Средняя длительность: {df['duration'].mean():.2f} часов")
print(f"  Медианная длительность: {df['duration'].median():.2f} часов")
print(f"  Минимальная: {df['duration'].min():.2f} часов")
print(f"  Максимальная: {df['duration'].max():.2f} часов")

print("\n💡 Рекомендации на основе анализа:")
if len(df[df['exit_reason'] == 'time_exit_12']) > len(df) * 0.5:
    print("  ⚠️ Больше 50% сделок закрываются по времени (time_exit_12)")
    print("     → Рассмотрите увеличение времени удержания позиции или улучшение TP/SL")
    
if len(df[df['exit_reason'] == 'SL_hit']) > len(df) * 0.3:
    print("  ⚠️ Более 30% сделок закрываются по Stop Loss")
    print("     → Ужесточите фильтры входа или увеличьте расстояние до SL")
    
if len(df[df['exit_reason'] == 'TP_hit']) < len(df) * 0.1:
    print("  ⚠️ Менее 10% сделок достигают Take Profit")
    print("     → Рассмотрите снижение TP или улучшение фильтров входа")
