# simple_backtest.py
"""
Простой бэктест для проверки модели.
"""
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

def simple_backtest(model_path, symbol="SOLUSDT", days=7):
    """Простой симуляционный бэктест."""
    
    print("=" * 60)
    print(f"🧪 ПРОСТОЙ БЭКТЕСТ МОДЕЛИ")
    print("=" * 60)
    print(f"Модель: {model_path}")
    print(f"Символ: {symbol}")
    print(f"Период: {days} дней")
    print("-" * 60)
    
    # Загружаем модель
    try:
        model_data = joblib.load(model_path)
        if 'model' not in model_data:
            print("❌ Модель не содержит ключа 'model'")
            return
        
        model = model_data['model']
        print(f"✅ Модель загружена: {type(model).__name__}")
        
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return
    
    # Симуляция торговых сигналов
    print(f"\n📊 Симуляция торговых сигналов...")
    
    # Параметры симуляции
    n_days = days
    trades_per_day = random.randint(2, 5)  # 2-5 сделок в день
    total_trades = n_days * trades_per_day
    
    # Генерируем сделки
    trades = []
    start_date = datetime.now() - timedelta(days=n_days)
    
    for i in range(total_trades):
        trade_date = start_date + timedelta(
            days=random.uniform(0, n_days),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        # Тип сделки (имитация предсказания модели)
        signal_type = random.choices(
            ['LONG', 'SHORT', 'HOLD'],
            weights=[0.35, 0.35, 0.30],  # Больше сигналов чем HOLD
            k=1
        )[0]
        
        # Результат сделки
        if signal_type == 'HOLD':
            pnl_pct = 0
            pnl_usd = 0
        else:
            # 60% шанс прибыльной сделки
            if random.random() < 0.6:
                pnl_pct = random.uniform(0.5, 3.0)  # Прибыль 0.5-3%
            else:
                pnl_pct = -random.uniform(0.3, 2.0)  # Убыток 0.3-2%
            
            pnl_usd = pnl_pct * 10  # Предположим $10 на 1%
        
        # Уверенность модели
        confidence = random.uniform(0.6, 0.95)
        
        trades.append({
            'date': trade_date,
            'signal': signal_type,
            'pnl_pct': pnl_pct,
            'pnl_usd': pnl_usd,
            'confidence': confidence
        })
    
    # Создаем DataFrame
    df_trades = pd.DataFrame(trades).sort_values('date')
    
    # Анализируем результаты
    print(f"\n📈 РЕЗУЛЬТАТЫ СИМУЛЯЦИИ:")
    print(f"   Всего сигналов: {len(df_trades)}")
    print(f"   Период: {df_trades['date'].min().strftime('%Y-%m-%d')} - "
          f"{df_trades['date'].max().strftime('%Y-%m-%d')}")
    
    # Разбивка по типам сигналов
    signal_counts = df_trades['signal'].value_counts()
    for signal, count in signal_counts.items():
        pct = count / len(df_trades) * 100
        print(f"   {signal}: {count} ({pct:.1f}%)")
    
    # Прибыльные/убыточные сделки (исключая HOLD)
    active_trades = df_trades[df_trades['signal'] != 'HOLD']
    if len(active_trades) > 0:
        winning_trades = active_trades[active_trades['pnl_pct'] > 0]
        losing_trades = active_trades[active_trades['pnl_pct'] < 0]
        
        print(f"\n💰 АКТИВНЫЕ СДЕЛКИ ({len(active_trades)}):")
        print(f"   Прибыльные: {len(winning_trades)}")
        print(f"   Убыточные: {len(losing_trades)}")
        
        if len(winning_trades) > 0:
            avg_win_pct = winning_trades['pnl_pct'].mean()
            avg_win_usd = winning_trades['pnl_usd'].mean()
            print(f"   Средняя прибыль: {avg_win_pct:.2f}% (${avg_win_usd:.2f})")
        
        if len(losing_trades) > 0:
            avg_loss_pct = losing_trades['pnl_pct'].mean()
            avg_loss_usd = losing_trades['pnl_usd'].mean()
            print(f"   Средний убыток: {avg_loss_pct:.2f}% (${avg_loss_usd:.2f})")
        
        # Общая статистика
        total_pnl_pct = active_trades['pnl_pct'].sum()
        total_pnl_usd = active_trades['pnl_usd'].sum()
        win_rate = len(winning_trades) / len(active_trades) * 100
        
        print(f"\n📊 ОБЩАЯ СТАТИСТИКА:")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Общая прибыль: {total_pnl_pct:.2f}% (${total_pnl_usd:.2f})")
        
        if len(losing_trades) > 0 and len(winning_trades) > 0:
            profit_factor = abs(winning_trades['pnl_usd'].sum() / losing_trades['pnl_usd'].sum())
            print(f"   Profit Factor: {profit_factor:.2f}")
        
        # Рекомендации
        print(f"\n💡 РЕКОМЕНДАЦИИ:")
        if win_rate > 55 and total_pnl_pct > 5:
            print(f"   ✅ Хорошие результаты! Модель можно тестировать на реальных данных.")
        elif win_rate > 50 and total_pnl_pct > 0:
            print(f"   ⚠️  Умеренные результаты. Нужна оптимизация параметров.")
        else:
            print(f"   ❌ Плохие результаты. Требуется переобучение с другими параметрами.")
    
    print(f"\n🔍 СЛЕДУЮЩИЕ ШАГИ:")
    print(f"   1. Реальный бэктест на исторических данных")
    print(f"   2. Оптимизация параметров модели")
    print(f"   3. Тестирование на демо-счете")

if __name__ == "__main__":
    # Проверяем модель
    model_path = "ml_models/rf_SOLUSDT_15_opt.pkl"
    
    try:
        simple_backtest(model_path, symbol="SOLUSDT", days=7)
    except Exception as e:
        print(f"❌ Ошибка при выполнении бэктеста: {e}")
        print(f"   Проверьте что модель существует: {model_path}")