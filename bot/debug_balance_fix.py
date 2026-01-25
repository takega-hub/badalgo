import pandas as pd
import numpy as np
import sys
import os

# Добавляем путь к модулям
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crypto_env_v16_rr2_enhanced import CryptoTradingEnvV16_RR2_Enhanced
from data_processor_enhanced import DataProcessorEnhanced

def test_balance_calculations():
    """
    Тестирование исправлений расчета баланса и net_worth
    """
    print("=" * 70)
    print("🔧 ТЕСТИРОВАНИЕ ИСПРАВЛЕНИЙ РАСЧЕТА БАЛАНСА")
    print("=" * 70)
    
    # 1. Создаем тестовые данные
    print("\n📊 Создание тестовых данных...")
    
    # Создаем простой DataFrame для тестирования
    dates = pd.date_range('2024-01-01', periods=500, freq='15min')
    np.random.seed(42)
    
    test_data = pd.DataFrame({
        'open': np.random.normal(28000, 500, 500),
        'high': np.random.normal(28100, 600, 500),
        'low': np.random.normal(27900, 600, 500),
        'close': np.random.normal(28000, 500, 500),
        'volume': np.random.normal(100, 20, 500),
    })
    
    # Добавляем ATR для простоты
    high_low = test_data['high'] - test_data['low']
    high_close = np.abs(test_data['high'] - test_data['close'].shift())
    low_close = np.abs(test_data['low'] - test_data['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    test_data['atr'] = ranges.max(axis=1).rolling(window=14).mean()
    
    # Добавляем необходимые колонки
    test_data['ma_fast'] = test_data['close'].rolling(window=10).mean()
    test_data['ma_slow'] = test_data['close'].rolling(window=30).mean()
    test_data['trend'] = (test_data['ma_fast'] - test_data['ma_slow']) / test_data['ma_slow']
    
    # Заполняем NaN
    test_data = test_data.ffill().bfill()
    
    # 2. Тестовые признаки
    obs_cols = ['close', 'atr', 'trend', 'ma_fast', 'ma_slow']
    
    # 3. Создаем среду с ОТЛАДОЧНЫМИ параметрами
    print("\n🎯 Создание тестовой среды...")
    
    env = CryptoTradingEnvV16_RR2_Enhanced(
        df=test_data,
        obs_cols=obs_cols,
        initial_balance=1000.0,
        commission=0.001,
        slippage=0.0005,
        log_file="debug_balance_test.csv",
        rr_ratio=2.0,
        atr_multiplier=2.0,  # Уменьшили для тестирования
        training_mode="rr2_enhanced"
    )
    
    # 4. Модифицируем параметры для тестирования
    env.tp_levels = [0.8, 1.2, 1.6]  # Более достижимые TP
    env.tp_close_percentages = [0.50, 0.30, 0.20]  # Больше на первом уровне
    env.trailing_activation_atr = 0.5  # Ранняя активация
    env.trailing_distance_atr = 0.8    # Ближе трейлинг
    
    # 5. Запускаем тестовые сделки
    print("\n🔍 Запуск тестовых сделок...")
    print("-" * 50)
    
    obs, _ = env.reset()
    
    test_scenarios = [
        # (шаги, действия) - симулируем разные сценарии
        (10, 0),   # Hold
        (5, 1),    # Open Long
        (10, 0),   # Hold position
        (5, 0),    # Wait for TP/SL
        (5, 2),    # Close long, open short (если еще открыт)
        (10, 0),   # Hold short
        (5, 1),    # Close short, open long
    ]
    
    total_steps = 0
    for steps, action in test_scenarios:
        for i in range(steps):
            obs, reward, terminated, truncated, info = env.step(action)
            total_steps += 1
            
            # Детальный лог каждые 3 шага
            if total_steps % 3 == 0:
                print(f"\n[Step {total_steps}] Action: {action}, Position: {env.position}")
                print(f"  Balance: {env.balance:.2f}, Net Worth: {env.net_worth:.2f}")
                print(f"  Unrealized PnL: {(env.net_worth - env.balance):.2f}")
                
                if env.position != 0:
                    current_price = env.df.loc[env.current_step, "close"]
                    if env.position == 1:
                        unrealized = (current_price - env.entry_price) / env.entry_price * 100
                    else:
                        unrealized = (env.entry_price - current_price) / env.entry_price * 100
                    print(f"  Current Price: {current_price:.2f}, Unrealized %: {unrealized:.2f}%")
            
            if terminated or truncated:
                print("⏹️  Среда завершена")
                break
        
        if terminated or truncated:
            break
    
    # 6. Анализ результатов
    print("\n" + "=" * 70)
    print("📈 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 70)
    
    # Основные метрики
    metrics = env.get_performance_metrics()
    
    print(f"\n💰 БАЛАНС:")
    print(f"  Начальный баланс: {env.initial_balance:.2f}")
    print(f"  Конечный баланс: {env.balance:.2f}")
    print(f"  Конечный Net Worth: {env.net_worth:.2f}")
    print(f"  Общая прибыль/убыток: {(env.balance - env.initial_balance):.2f} ({((env.balance/env.initial_balance - 1) * 100):.2f}%)")
    
    # Проверка согласованности
    print(f"\n🔍 ПРОВЕРКА РАСЧЕТОВ:")
    
    # 1. Проверка net_worth vs balance при отсутствии позиции
    if env.position == 0:
        diff = abs(env.net_worth - env.balance)
        if diff < 0.01:
            print(f"  ✅ Net Worth и Balance совпадают: diff = {diff:.6f}")
        else:
            print(f"  ⚠️  Несоответствие! Net Worth ({env.net_worth:.2f}) ≠ Balance ({env.balance:.2f})")
            print(f"     Разница: {diff:.2f}")
    
    # 2. Проверка истории сделок
    print(f"\n📊 СДЕЛКИ (всего {env.total_trades}):")
    if env.trade_history:
        for i, trade in enumerate(env.trade_history[-5:], 1):  # Последние 5 сделок
            print(f"  Сделка {i}:")
            print(f"    Тип: {trade['type']}, Выход: {trade['exit_type']}")
            print(f"    Entry: {trade['entry_price']:.2f}, Exit: {trade['exit_price']:.2f}")
            print(f"    PnL: {trade['pnl']*100:.2f}%, Длительность: {trade['duration']} шагов")
            print(f"    Баланс до/после: {trade.get('balance', 'N/A')}")
    
    # 3. Статистика по типам выходов
    print(f"\n🎯 СТАТИСТИКА ВЫХОДОВ:")
    exit_stats = env._get_info().get('exit_stats', {})
    for exit_type, count in exit_stats.items():
        if count > 0:
            print(f"  {exit_type}: {count} сделок")
    
    # 4. Проверка частичных закрытий
    print(f"\n🎯 ЧАСТИЧНЫЕ ЗАКРЫТИЯ:")
    tp_closed_stats = env._get_info().get('tp_level_stats', {})
    for level, count in tp_closed_stats.items():
        print(f"  {level}: {count} раз")
    
    # 5. Детальный анализ PnL
    print(f"\n📈 ДЕТАЛЬНЫЙ АНАЛИЗ PnL:")
    print(f"  Общий PnL: {env.total_pnl*100:.2f}%")
    print(f"  Выигрыши: {env.winning_trades}, Проигрыши: {env.losing_trades}")
    print(f"  Винрейт: {(env.winning_trades/max(1, env.total_trades)*100):.1f}%")
    
    # 6. Расчет Profit Factor
    if env.winning_trades > 0 and env.losing_trades > 0:
        winning_pnls = [t['pnl'] for t in env.trade_history if t['pnl'] > 0]
        losing_pnls = [abs(t['pnl']) for t in env.trade_history if t['pnl'] < 0]
        
        avg_win = np.mean(winning_pnls) if winning_pnls else 0
        avg_loss = np.mean(losing_pnls) if losing_pnls else 0
        
        if avg_loss > 0:
            profit_factor = avg_win / avg_loss
            print(f"  Profit Factor: {profit_factor:.2f}")
        else:
            print(f"  Profit Factor: ∞ (нет убытков)")
    
    # 7. Сохранение логов для анализа
    print(f"\n💾 Логи сохранены в: debug_balance_test.csv")
    
    # 8. Рекомендации по настройке
    print(f"\n🎯 РЕКОМЕНДАЦИИ ПО НАСТРОЙКЕ:")
    
    # Анализ эффективности TP
    total_tp = sum([count for exit_type, count in exit_stats.items() 
                    if 'TP' in exit_type])
    tp_rate = (total_tp / max(1, env.total_trades)) * 100
    
    print(f"  1. TP Rate: {tp_rate:.1f}% - {'ХОРОШО' if tp_rate > 30 else 'МОЖНО ЛУЧШЕ'}")
    print(f"     • Увеличить tp_bonus_multiplier с {env.tp_bonus_multiplier} до {env.tp_bonus_multiplier * 1.5}")
    
    # Анализ SL
    total_sl = sum([count for exit_type, count in exit_stats.items() 
                    if 'SL' in exit_type])
    sl_rate = (total_sl / max(1, env.total_trades)) * 100
    
    print(f"  2. SL Rate: {sl_rate:.1f}% - {'НОРМАЛЬНО' if sl_rate < 70 else 'СЛИШКОМ МНОГО'}")
    if sl_rate > 70:
        print(f"     • Уменьшить atr_multiplier с {env.atr_multiplier} до {env.atr_multiplier * 0.8}")
        print(f"     • Уменьшить max_sl_percent с {env.max_sl_percent*100}% до {env.max_sl_percent*100*0.8}%")
    
    # Анализ MANUAL
    manual_count = exit_stats.get('MANUAL', 0)
    manual_rate = (manual_count / max(1, env.total_trades)) * 100
    
    print(f"  3. MANUAL Rate: {manual_rate:.1f}% - {'ОТЛИЧНО' if manual_rate < 10 else 'МОЖНО ЛУЧШЕ'}")
    if manual_rate > 10:
        print(f"     • Увеличить manual_penalty с {env.manual_penalty} до {env.manual_penalty * 2}")
        print(f"     • Уменьшить max_hold_steps с {env.max_hold_steps} до {env.max_hold_steps * 0.7}")
    
    return env

def run_quick_backtest():
    """
    Быстрый бэктест с исправленными расчетами
    """
    print("\n" + "=" * 70)
    print("🚀 БЫСТРЫЙ БЭКТЕСТ НА РЕАЛЬНЫХ ДАННЫХ")
    print("=" * 70)
    
    try:
        # Загружаем реальные данные
        data_path = "data/btc_15m.csv"
        processor = DataProcessorEnhanced(data_path)
        processor.load_data()
        df = processor.prepare_features()
        
        # Берем небольшой участок для теста
        test_df = df.head(1000).reset_index(drop=True)
        
        # Простые признаки для теста
        simple_cols = ['close', 'atr', 'rsi_norm', 'trend_bias_1h', 'volatility_ratio']
        simple_cols = [col for col in simple_cols if col in test_df.columns]
        
        print(f"\n📊 Данные загружены: {len(test_df)} строк")
        print(f"📈 Признаков: {len(simple_cols)}")
        
        # Создаем среду
        env = CryptoTradingEnvV16_RR2_Enhanced(
            df=test_df,
            obs_cols=simple_cols,
            initial_balance=1000,
            commission=0.001,
            slippage=0.0005,
            log_file="quick_backtest.csv",
            rr_ratio=2.0,
            atr_multiplier=2.5,
            training_mode="rr2_enhanced"
        )
        
        # Запускаем случайную стратегию для теста
        print("\n🎯 Запуск бэктеста...")
        
        obs, _ = env.reset()
        done = False
        step = 0
        
        while not done and step < 500:
            # Случайное действие (0=hold, 1=long, 2=short)
            action = np.random.choice([0, 1, 2], p=[0.7, 0.15, 0.15])
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            
            # Лог каждые 50 шагов
            if step % 50 == 0:
                print(f"  Step {step}: Balance: {env.balance:.2f}, Net Worth: {env.net_worth:.2f}, "
                      f"Trades: {env.total_trades}")
        
        # Результаты
        print(f"\n📊 РЕЗУЛЬТАТЫ БЭКТЕСТА:")
        print(f"  Итоговый баланс: {env.balance:.2f}")
        print(f"  Сделок: {env.total_trades}")
        
        if env.total_trades > 0:
            win_rate = (env.winning_trades / env.total_trades) * 100
            print(f"  Винрейт: {win_rate:.1f}%")
            
            # Анализ типов выходов
            exit_stats = env._get_info().get('exit_stats', {})
            print(f"  Типы выходов:")
            for exit_type, count in exit_stats.items():
                if count > 0:
                    percentage = (count / env.total_trades) * 100
                    print(f"    {exit_type}: {count} ({percentage:.1f}%)")
        
        print(f"\n💾 Подробные логи в: quick_backtest.csv")
        
    except Exception as e:
        print(f"\n❌ Ошибка при загрузке данных: {e}")
        print("  Используется генерация тестовых данных...")
        test_balance_calculations()

if __name__ == "__main__":
    print("🚀 ЗАПУСК ТЕСТИРОВАНИЯ ИСПРАВЛЕНИЙ БАЛАНСА")
    print("=" * 70)
    
    # Запускаем тесты
    env = test_balance_calculations()
    
    # Дополнительный быстрый бэктест
    run_quick_backtest()
    
    print("\n" + "=" * 70)
    print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 70)
    
    # Инструкции
    print("\n🎯 СЛЕДУЮЩИЕ ШАГИ:")
    print("1. Проверьте файл debug_balance_test.csv на согласованность расчетов")
    print("2. Если есть расхождения баланса - проверьте методы _close_position и _partial_close")
    print("3. Отрегулируйте параметры TP/SL на основе рекомендаций выше")
    print("4. Запустите полное обучение: python train_v16_rr2_enhanced.py")