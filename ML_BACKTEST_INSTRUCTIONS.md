# 📊 Инструкция по бэктестингу ML стратегий

## 🚀 Быстрый старт

### 1. Бэктест одной модели:

```bash
python backtest_ml_strategy.py --model ml_models/triple_ensemble_ETHUSDT_15.pkl --symbol ETHUSDT --days 30
```

### 2. Бэктест всех моделей для символа:

```bash
# BTCUSDT
python backtest_ml_strategy.py --model ml_models/triple_ensemble_BTCUSDT_15.pkl --symbol BTCUSDT --days 30
python backtest_ml_strategy.py --model ml_models/ensemble_BTCUSDT_15.pkl --symbol BTCUSDT --days 30

# ETHUSDT
python backtest_ml_strategy.py --model ml_models/triple_ensemble_ETHUSDT_15.pkl --symbol ETHUSDT --days 30
python backtest_ml_strategy.py --model ml_models/ensemble_ETHUSDT_15.pkl --symbol ETHUSDT --days 30

# SOLUSDT
python backtest_ml_strategy.py --model ml_models/triple_ensemble_SOLUSDT_15.pkl --symbol SOLUSDT --days 30
python backtest_ml_strategy.py --model ml_models/ensemble_SOLUSDT_15.pkl --symbol SOLUSDT --days 30
```

### 3. С кастомными параметрами:

```bash
python backtest_ml_strategy.py \
    --model ml_models/triple_ensemble_ETHUSDT_15.pkl \
    --symbol ETHUSDT \
    --days 30 \
    --balance 1000 \
    --risk 0.02 \
    --leverage 10
```

## 📊 Параметры

- `--model`: Путь к ML модели (обязательно)
- `--symbol`: Торговая пара (по умолчанию: BTCUSDT)
- `--days`: Количество дней для бэктеста (по умолчанию: 30)
- `--interval`: Таймфрейм (по умолчанию: 15m)
- `--balance`: Начальный баланс в USD (по умолчанию: 1000)
- `--risk`: Риск на сделку как доля (по умолчанию: 0.02 = 2%)
- `--leverage`: Плечо (по умолчанию: 10)

## 📈 Метрики в результатах

Скрипт рассчитывает:

1. **Финансовые метрики:**
   - Total PnL (общая прибыль/убыток)
   - Max Drawdown (максимальная просадка)
   - Final Balance (финальный баланс)

2. **Статистика сделок:**
   - Total Trades (всего сделок)
   - Win Rate (процент прибыльных сделок)
   - Profit Factor (отношение прибыли к убыткам)
   - Sharpe Ratio (коэффициент Шарпа)

3. **Детали сделок:**
   - Average Win/Loss (средняя прибыль/убыток)
   - Best/Worst Trade (лучшая/худшая сделка)
   - Consecutive Wins/Losses (серии побед/поражений)

4. **Распределение сигналов:**
   - LONG/SHORT сигналы
   - Average Confidence (средняя уверенность модели)

## 📁 Результаты

Результаты сохраняются в файл:
```
ml_backtest_{SYMBOL}_{MODEL_NAME}_{TIMESTAMP}.txt
```

Пример: `ml_backtest_ETHUSDT_triple_ensemble_ETHUSDT_15_20250127_123456.txt`

## 🔍 Сравнение моделей

Для сравнения разных моделей запустите бэктест для каждой и сравните метрики:

```bash
# Triple Ensemble
python backtest_ml_strategy.py --model ml_models/triple_ensemble_ETHUSDT_15.pkl --symbol ETHUSDT --days 30

# Стандартный Ensemble
python backtest_ml_strategy.py --model ml_models/ensemble_ETHUSDT_15.pkl --symbol ETHUSDT --days 30

# LightGBM (если есть)
python backtest_ml_strategy.py --model ml_models/lgb_ETHUSDT_15.pkl --symbol ETHUSDT --days 30
```

## ⚙️ Настройки риска

Рекомендуемые параметры:

- **Консервативный:** `--risk 0.01 --leverage 5`
- **Умеренный:** `--risk 0.02 --leverage 10` (по умолчанию)
- **Агрессивный:** `--risk 0.03 --leverage 15`

## 📊 Интерпретация результатов

### Хорошие результаты:
- ✅ Win Rate > 50%
- ✅ Profit Factor > 1.5
- ✅ Total PnL > 0
- ✅ Max Drawdown < 20%

### Требуют улучшения:
- ⚠️ Win Rate < 45%
- ⚠️ Profit Factor < 1.2
- ⚠️ Max Drawdown > 30%

### Плохие результаты:
- ❌ Total PnL < 0
- ❌ Profit Factor < 1.0
- ❌ Win Rate < 40%

## 🎯 Следующие шаги

1. Запустите бэктест для всех моделей
2. Сравните результаты
3. Выберите лучшую модель
4. Протестируйте на более длинном периоде (60-90 дней)
5. Оптимизируйте параметры (risk, leverage)
