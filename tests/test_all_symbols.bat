@echo off
echo ================================================================================
echo Testing all strategies on ETHUSDT and SOLUSDT
echo ================================================================================

echo.
echo Testing ETHUSDT...
echo.

echo --- TREND Strategy on ETHUSDT ---
python test_all_strategies.py --strategy trend --symbol ETHUSDT --days 30

echo.
echo --- FLAT Strategy on ETHUSDT ---
python test_all_strategies.py --strategy flat --symbol ETHUSDT --days 30

echo.
echo --- MOMENTUM Strategy on ETHUSDT ---
python test_all_strategies.py --strategy momentum --symbol ETHUSDT --days 30

echo.
echo --- LIQUIDITY Strategy on ETHUSDT ---
python test_all_strategies.py --strategy liquidity --symbol ETHUSDT --days 30

echo.
echo --- SMC Strategy on ETHUSDT ---
python test_all_strategies.py --strategy smc --symbol ETHUSDT --days 30

echo.
echo --- ICT Strategy on ETHUSDT ---
python test_all_strategies.py --strategy ict --symbol ETHUSDT --days 30

echo.
echo --- ML Strategy on ETHUSDT ---
python test_all_strategies.py --strategy ml --symbol ETHUSDT --days 30

echo.
echo ================================================================================
echo Testing SOLUSDT...
echo.

echo --- TREND Strategy on SOLUSDT ---
python test_all_strategies.py --strategy trend --symbol SOLUSDT --days 30

echo.
echo --- FLAT Strategy on SOLUSDT ---
python test_all_strategies.py --strategy flat --symbol SOLUSDT --days 30

echo.
echo --- MOMENTUM Strategy on SOLUSDT ---
python test_all_strategies.py --strategy momentum --symbol SOLUSDT --days 30

echo.
echo --- LIQUIDITY Strategy on SOLUSDT ---
python test_all_strategies.py --strategy liquidity --symbol SOLUSDT --days 30

echo.
echo --- SMC Strategy on SOLUSDT ---
python test_all_strategies.py --strategy smc --symbol SOLUSDT --days 30

echo.
echo --- ICT Strategy on SOLUSDT ---
python test_all_strategies.py --strategy ict --symbol SOLUSDT --days 30

echo.
echo --- ML Strategy on SOLUSDT ---
python test_all_strategies.py --strategy ml --symbol SOLUSDT --days 30

echo.
echo ================================================================================
echo All tests completed!
echo ================================================================================
pause
