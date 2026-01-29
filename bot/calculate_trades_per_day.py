"""Скрипт для расчета количества сделок в день"""
import pandas as pd
import sys
import os

log_file = './logs/v18_mtf/train_v18_mtf_log.csv'

if not os.path.exists(log_file):
    print(f"Файл {log_file} не найден!")
    sys.exit(1)

# Загружаем данные
df = pd.read_csv(log_file, header=None, names=[
    'step', 'type', 'entry', 'sl_initial', 'sl_current', 'tp_levels', 
    'exit', 'pnl_percent', 'net_worth', 'exit_reason', 'duration', 
    'trailing', 'tp_closed', 'partial_closes', 'trade_quality', 'rr_ratio'
])

total_trades = len(df)
steps = df['step'].astype(int)
total_steps = steps.max() - steps.min()

# Для 15m таймфрейма: 96 шагов = 1 день (24 часа * 4 свечи в час)
days = total_steps / 96 if total_steps > 0 else 1
trades_per_day = total_trades / days if days > 0 else 0

print(f"📊 АНАЛИЗ КОЛИЧЕСТВА СДЕЛОК")
print(f"="*50)
print(f"Всего сделок: {total_trades}")
print(f"Общий период (шагов): {total_steps}")
print(f"Оценка дней (96 шагов/день для 15m): {days:.1f}")
print(f"Сделок в день: {trades_per_day:.2f}")
print(f"="*50)
print(f"🎯 Целевой показатель: 1-10 сделок в день")
if 1 <= trades_per_day <= 10:
    print(f"✅ Отлично! Текущий показатель ({trades_per_day:.2f}) в целевом диапазоне")
elif trades_per_day > 10:
    print(f"⚠️  Слишком много сделок ({trades_per_day:.2f}). Нужно ужесточить фильтры.")
else:
    print(f"⚠️  Слишком мало сделок ({trades_per_day:.2f}). Возможно, фильтры слишком строгие.")
