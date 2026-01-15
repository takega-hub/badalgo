# Команды для тестирования всех стратегий на ETHUSDT и SOLUSDT

## ETHUSDT

```bash
python test_all_strategies.py --strategy trend --symbol ETHUSDT --days 30
python test_all_strategies.py --strategy flat --symbol ETHUSDT --days 30
python test_all_strategies.py --strategy momentum --symbol ETHUSDT --days 30
python test_all_strategies.py --strategy liquidity --symbol ETHUSDT --days 30
python test_all_strategies.py --strategy smc --symbol ETHUSDT --days 30
python test_all_strategies.py --strategy ict --symbol ETHUSDT --days 30
python test_all_strategies.py --strategy ml --symbol ETHUSDT --days 30
```

## SOLUSDT

```bash
python test_all_strategies.py --strategy trend --symbol SOLUSDT --days 30
python test_all_strategies.py --strategy flat --symbol SOLUSDT --days 30
python test_all_strategies.py --strategy momentum --symbol SOLUSDT --days 30
python test_all_strategies.py --strategy liquidity --symbol SOLUSDT --days 30
python test_all_strategies.py --strategy smc --symbol SOLUSDT --days 30
python test_all_strategies.py --strategy ict --symbol SOLUSDT --days 30
python test_all_strategies.py --strategy ml --symbol SOLUSDT --days 30
```

## Или используйте bat-файл:

```bash
test_all_symbols.bat
```
