import pandas as pd

df = pd.read_csv('logs/v17_optimized_v2/train_v17_log.csv')

# 96 шагов = 1 день (15-минутные свечи: 96 * 15 мин = 1440 мин = 24 часа)
df['day'] = df['step'] // 96

# Подсчет сделок по дням (исключаем частичные закрытия для подсчета уникальных сделок)
unique_trades = df[~df['type'].str.contains('PARTIAL', na=False)]
trades_per_day = unique_trades.groupby('day').size()

print('=' * 60)
print('СТАТИСТИКА СДЕЛОК ПО ДНЯМ')
print('=' * 60)
print(f'\nВсего дней: {len(trades_per_day)}')
print(f'Всего уникальных сделок: {len(unique_trades)}')
print(f'\nСреднее: {trades_per_day.mean():.2f} сделок/день')
print(f'Медиана: {trades_per_day.median():.2f} сделок/день')
print(f'Максимум: {trades_per_day.max()} сделок/день')
print(f'Минимум: {trades_per_day.min()} сделок/день')
print(f'\nЛимит в коде: max_daily_trades = 5 (но в train_v17_optimized.py установлен 15)')

print('\n' + '=' * 60)
print('РАСПРЕДЕЛЕНИЕ ПО ДНЯМ:')
print('=' * 60)
for day, count in trades_per_day.items():
    print(f'День {day:2d}: {count:2d} сделок')
