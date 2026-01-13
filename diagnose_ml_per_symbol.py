#!/usr/bin/env python3
"""
Диагностика ML стратегии для всех символов.
Проверяет почему сигналы могут не генерироваться для некоторых пар.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

from bot.config import load_settings
from bot.exchange.bybit_client import BybitClient
from bot.indicators import prepare_with_indicators
from bot.ml.strategy_ml import MLStrategy, build_ml_signals
from bot.strategy import Action
import pandas as pd
import numpy as np


def diagnose_ml_for_symbol(symbol: str, settings):
    """Диагностирует ML стратегию для конкретного символа."""
    print(f"\n{'='*60}")
    print(f"📊 ДИАГНОСТИКА ML ДЛЯ {symbol}")
    print(f"{'='*60}")
    
    # 1. Проверяем наличие модели
    models_dir = Path(__file__).parent / "ml_models"
    model_files = list(models_dir.glob(f"*_{symbol}_*.pkl"))
    
    print(f"\n1️⃣ МОДЕЛИ:")
    if not model_files:
        print(f"   ❌ Модели для {symbol} НЕ НАЙДЕНЫ!")
        return
    
    for mf in model_files:
        print(f"   ✅ {mf.name}")
    
    # Выбираем ensemble модель (предпочтительную)
    ensemble_model = None
    for mf in model_files:
        if "ensemble" in mf.name.lower():
            ensemble_model = mf
            break
    
    if not ensemble_model:
        ensemble_model = model_files[0]  # Берём первую доступную
    
    print(f"   📍 Используем: {ensemble_model.name}")
    
    # 2. Загружаем данные
    print(f"\n2️⃣ ДАННЫЕ:")
    client = BybitClient(settings.api)
    
    try:
        df = client.get_klines(symbol=symbol, interval="15", limit=500)
        print(f"   ✅ Загружено {len(df)} свечей")
        print(f"   📅 Период: {df.index[0]} - {df.index[-1]}")
    except Exception as e:
        print(f"   ❌ Ошибка загрузки данных: {e}")
        return
    
    # 3. Подготавливаем индикаторы
    print(f"\n3️⃣ ИНДИКАТОРЫ:")
    try:
        df_ready = prepare_with_indicators(
            df,
            settings.strategy.sma_length,
            settings.strategy.rsi_length,
            settings.strategy.breakout_lookback,
        )
        print(f"   ✅ Подготовлено {len(df_ready)} строк")
    except Exception as e:
        print(f"   ❌ Ошибка подготовки индикаторов: {e}")
        return
    
    # 4. Загружаем и проверяем модель
    print(f"\n4️⃣ МОДЕЛЬ:")
    try:
        strategy = MLStrategy(
            str(ensemble_model),
            confidence_threshold=settings.ml_confidence_threshold,
            min_signal_strength=settings.ml_min_signal_strength,
            stability_filter=settings.ml_stability_filter,
        )
        print(f"   ✅ Модель загружена успешно")
        
        # Метаданные модели
        metadata = strategy.model_data.get("metadata", {})
        print(f"   📍 Тип модели: {metadata.get('model_type', 'unknown')}")
        print(f"   📅 Дата обучения: {metadata.get('trained_at', 'unknown')}")
        print(f"   📊 CV Accuracy: {metadata.get('cv_mean', 0):.4f}")
        print(f"   📊 F1 Score: {metadata.get('f1_score', 0):.4f}")
        
        # Data info
        data_info = strategy.model_data.get("data_info", {})
        if data_info:
            print(f"   📊 Распределение классов при обучении: {data_info}")
    except Exception as e:
        print(f"   ❌ Ошибка загрузки модели: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. Генерируем сигналы
    print(f"\n5️⃣ ГЕНЕРАЦИЯ СИГНАЛОВ:")
    try:
        signals = build_ml_signals(
            df_ready,
            str(ensemble_model),
            settings.ml_confidence_threshold,
            settings.ml_min_signal_strength,
            settings.ml_stability_filter,
        )
        
        total_signals = len(signals)
        long_signals = [s for s in signals if s.action == Action.LONG]
        short_signals = [s for s in signals if s.action == Action.SHORT]
        hold_signals = [s for s in signals if s.action == Action.HOLD]
        
        print(f"   📊 Всего сигналов: {total_signals}")
        print(f"   🟢 LONG: {len(long_signals)}")
        print(f"   🔴 SHORT: {len(short_signals)}")
        print(f"   ⚪ HOLD: {len(hold_signals)}")
        
        if long_signals or short_signals:
            print(f"\n   📈 Последние actionable сигналы:")
            actionable = (long_signals + short_signals)[-5:]
            for sig in actionable:
                ts = sig.timestamp.strftime('%Y-%m-%d %H:%M') if hasattr(sig.timestamp, 'strftime') else str(sig.timestamp)
                print(f"      {sig.action.value} @ ${sig.price:.2f} [{ts}]")
                print(f"         Причина: {sig.reason[:80]}...")
        else:
            print(f"\n   ⚠️ НЕТ actionable сигналов (LONG/SHORT)!")
    except Exception as e:
        print(f"   ❌ Ошибка генерации сигналов: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 6. Проверяем предсказания напрямую
    print(f"\n6️⃣ ПРЯМЫЕ ПРЕДСКАЗАНИЯ (последние 10 свечей):")
    try:
        # Берём последние 10 строк и делаем предсказания
        last_rows = df_ready.tail(10)
        
        for idx, row in last_rows.iterrows():
            try:
                df_single = df_ready.loc[:idx].tail(50)  # Нужен контекст для фичей
                if len(df_single) < 20:
                    continue
                
                prediction, confidence = strategy.predict(df_single, skip_feature_creation=False)
                
                action_map = {-1: "SHORT", 0: "HOLD", 1: "LONG"}
                action_str = action_map.get(prediction, "UNKNOWN")
                
                # Определяем, прошёл бы сигнал фильтры
                passes_threshold = confidence >= settings.ml_confidence_threshold
                
                ts = idx.strftime('%H:%M') if hasattr(idx, 'strftime') else str(idx)[-5:]
                status = "✅" if passes_threshold and prediction != 0 else "❌"
                
                print(f"   {ts}: {action_str} conf={confidence:.2%} {status}")
                
            except Exception as e:
                continue
    except Exception as e:
        print(f"   ❌ Ошибка проверки предсказаний: {e}")
    
    # 7. Анализ фильтров
    print(f"\n7️⃣ ПАРАМЕТРЫ ФИЛЬТРАЦИИ:")
    print(f"   📍 Confidence threshold: {settings.ml_confidence_threshold}")
    print(f"   📍 Min signal strength: {settings.ml_min_signal_strength}")
    print(f"   📍 Stability filter: {settings.ml_stability_filter}")


def main():
    print("=" * 60)
    print("🔍 ДИАГНОСТИКА ML СТРАТЕГИИ ПО ВСЕМ СИМВОЛАМ")
    print("=" * 60)
    
    # Загружаем настройки
    settings = load_settings()
    
    print(f"\n📋 ГЛОБАЛЬНЫЕ НАСТРОЙКИ:")
    print(f"   ML Enabled: {settings.enable_ml_strategy}")
    print(f"   ML Model Path: {settings.ml_model_path}")
    print(f"   ML Confidence Threshold: {settings.ml_confidence_threshold}")
    print(f"   ML Min Signal Strength: {settings.ml_min_signal_strength}")
    print(f"   ML Stability Filter: {settings.ml_stability_filter}")
    
    # Проверяем каждый символ
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    
    for symbol in symbols:
        diagnose_ml_for_symbol(symbol, settings)
    
    print(f"\n{'='*60}")
    print("✅ ДИАГНОСТИКА ЗАВЕРШЕНА")
    print("=" * 60)


if __name__ == "__main__":
    main()
