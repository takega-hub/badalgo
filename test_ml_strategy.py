"""
Скрипт для тестирования ML стратегии на исторических данных.
Показывает, сколько и какие сигналы генерирует стратегия.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

from bot.config import load_settings
from bot.exchange.bybit_client import BybitClient
from bot.indicators import prepare_with_indicators
from bot.strategy import enrich_for_strategy
from bot.ml.strategy_ml import build_ml_signals
from bot.strategy import Action


def test_ml_strategy(
    symbol: str = "BTCUSDT",
    days_back: int = 7,
    model_path: str = None,
    confidence_threshold: float = 0.7,
    min_signal_strength: str = "умеренное",
    stability_filter: bool = True,
):
    """
    Тестирует ML стратегию на исторических данных.
    
    Args:
        symbol: Торговая пара
        days_back: Количество дней назад для тестирования
        model_path: Путь к модели (если None, ищет автоматически)
        confidence_threshold: Порог уверенности
        min_signal_strength: Минимальная сила сигнала
        stability_filter: Использовать фильтр стабильности
    """
    print("=" * 80)
    print(f"🧪 Тестирование ML стратегии для {symbol}")
    print("=" * 80)
    
    # Загружаем настройки
    settings = load_settings()
    
    # Ищем модель, если путь не указан
    if model_path is None:
        ml_models_dir = Path(__file__).parent / "ml_models"
        # Ищем ансамбль для символа
        ensemble_model = ml_models_dir / f"ensemble_{symbol}_15.pkl"
        rf_model = ml_models_dir / f"rf_{symbol}_15.pkl"
        xgb_model = ml_models_dir / f"xgb_{symbol}_15.pkl"
        
        if ensemble_model.exists():
            model_path = str(ensemble_model)
            print(f"📦 Используется модель: {ensemble_model.name}")
        elif rf_model.exists():
            model_path = str(rf_model)
            print(f"📦 Используется модель: {rf_model.name}")
        elif xgb_model.exists():
            model_path = str(xgb_model)
            print(f"📦 Используется модель: {xgb_model.name}")
        else:
            print(f"❌ Модель для {symbol} не найдена!")
            print(f"   Искал в: {ml_models_dir}")
            print(f"   Ожидаемые файлы:")
            print(f"     - ensemble_{symbol}_15.pkl")
            print(f"     - rf_{symbol}_15.pkl")
            print(f"     - xgb_{symbol}_15.pkl")
            return
    
    if not Path(model_path).exists():
        print(f"❌ Модель не найдена: {model_path}")
        return
    
    print(f"📦 Модель: {Path(model_path).name}")
    print(f"⚙️  Параметры:")
    print(f"   - Confidence threshold: {confidence_threshold}")
    print(f"   - Min signal strength: {min_signal_strength}")
    print(f"   - Stability filter: {stability_filter}")
    print()
    
    # Получаем исторические данные
    print(f"📊 Собираем данные за последние {days_back} дней...")
    client = BybitClient(settings.api)
    
    # Вычисляем количество свечей (15 минут)
    # 1 день = 24 часа * 4 свечи в час = 96 свечей
    limit = days_back * 96 + 200  # +200 для индикаторов
    
    df_raw = client.get_kline_df(symbol=symbol, interval="15", limit=limit)
    
    if df_raw.empty:
        print(f"❌ Не удалось получить данные для {symbol}")
        return
    
    print(f"✅ Получено {len(df_raw)} свечей")
    print(f"   Период: {df_raw.index[0]} - {df_raw.index[-1]}")
    print()
    
    # Подготавливаем данные с индикаторами
    print("🔧 Вычисляем индикаторы...")
    df_ind = prepare_with_indicators(
        df_raw,
        adx_length=settings.strategy.adx_length,
        di_length=settings.strategy.di_length,
        sma_length=settings.strategy.sma_length,
        rsi_length=settings.strategy.rsi_length,
        breakout_lookback=settings.strategy.breakout_lookback,
        bb_length=settings.strategy.bb_length,
        bb_std=settings.strategy.bb_std,
        atr_length=14,
        ema_fast_length=settings.strategy.ema_fast_length,
        ema_slow_length=settings.strategy.ema_slow_length,
        ema_timeframe=settings.strategy.momentum_ema_timeframe,
    )
    
    df_ready = enrich_for_strategy(df_ind, settings.strategy)
    
    print(f"✅ Данные подготовлены: {len(df_ready)} свечей")
    print()
    
    # Генерируем сигналы
    print("🤖 Генерируем ML сигналы...")
    print("-" * 80)
    
    try:
        signals = build_ml_signals(
            df_ready,
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            min_signal_strength=min_signal_strength,
            stability_filter=stability_filter,
            leverage=settings.leverage,
            target_profit_pct_margin=settings.ml_target_profit_pct_margin,
            max_loss_pct_margin=settings.ml_max_loss_pct_margin,
        )
        
        print(f"✅ Сгенерировано {len(signals)} сигналов")
        print()
        
        # Анализируем сигналы
        print("=" * 80)
        print("📈 АНАЛИЗ СИГНАЛОВ")
        print("=" * 80)
        
        if not signals:
            print("⚠️  НЕ СГЕНЕРИРОВАНО НИ ОДНОГО СИГНАЛА!")
            print()
            print("🔍 ДИАГНОСТИКА: Проверяем, почему модель не генерирует сигналы...")
            print()
            
            # Загружаем стратегию для диагностики
            from bot.ml.strategy_ml import MLStrategy
            strategy = MLStrategy(model_path, confidence_threshold, min_signal_strength, stability_filter)
            
            # Проверяем последние 50 баров
            diagnostic_bars = min(50, len(df_ready))
            print(f"📊 Анализируем последние {diagnostic_bars} баров...")
            print()
            
            predictions_stats = {"LONG": 0, "SHORT": 0, "HOLD": 0}
            confidence_stats = []
            filtered_reasons = {}
            
            for i in range(len(df_ready) - diagnostic_bars, len(df_ready)):
                try:
                    row = df_ready.iloc[i]
                    prediction, confidence = strategy.predict(df_ready.iloc[:i+1], skip_feature_creation=(i > len(df_ready) - diagnostic_bars + 10))
                    
                    if prediction == 1:
                        predictions_stats["LONG"] += 1
                    elif prediction == -1:
                        predictions_stats["SHORT"] += 1
                    else:
                        predictions_stats["HOLD"] += 1
                    
                    if prediction != 0:
                        confidence_stats.append(confidence)
                    
                    # Пробуем сгенерировать сигнал для диагностики
                    try:
                        from bot.strategy import Bias
                        # Используем df_ready для генерации сигнала
                        df_for_signal = df_ready.iloc[:i+1]
                        test_signal = strategy.generate_signal(
                            row=row,
                            df=df_for_signal,
                            has_position=None,
                            current_price=row["close"],
                            leverage=10,
                            target_profit_pct_margin=25.0,
                            max_loss_pct_margin=10.0,
                        )
                        if test_signal.action == Action.HOLD and test_signal.reason.startswith("ml_"):
                            # Извлекаем причину фильтрации
                            reason = test_signal.reason
                            if "сила_слишком_слабая" in reason:
                                filtered_reasons["сила_слишком_слабая"] = filtered_reasons.get("сила_слишком_слабая", 0) + 1
                            elif "ожидание" in reason:
                                filtered_reasons["ожидание_порог"] = filtered_reasons.get("ожидание_порог", 0) + 1
                            elif "индикаторы_не_согласны" in reason:
                                filtered_reasons["индикаторы_не_согласны"] = filtered_reasons.get("индикаторы_не_согласны", 0) + 1
                            elif "объем_не_подтверждает" in reason:
                                filtered_reasons["объем_не_подтверждает"] = filtered_reasons.get("объем_не_подтверждает", 0) + 1
                            elif "слабый_тренд" in reason:
                                filtered_reasons["слабый_тренд"] = filtered_reasons.get("слабый_тренд", 0) + 1
                            elif "экстремальная_зона" in reason:
                                filtered_reasons["экстремальная_зона"] = filtered_reasons.get("экстремальная_зона", 0) + 1
                    except:
                        pass
                except Exception as e:
                    continue
            
            print(f"📊 Статистика предсказаний модели (последние {diagnostic_bars} баров):")
            print(f"   LONG:  {predictions_stats['LONG']:3d} ({predictions_stats['LONG']/diagnostic_bars*100:.1f}%)")
            print(f"   SHORT: {predictions_stats['SHORT']:3d} ({predictions_stats['SHORT']/diagnostic_bars*100:.1f}%)")
            print(f"   HOLD:  {predictions_stats['HOLD']:3d} ({predictions_stats['HOLD']/diagnostic_bars*100:.1f}%)")
            print()
            
            if confidence_stats:
                print(f"📊 Статистика уверенности (для {len(confidence_stats)} actionable предсказаний):")
                print(f"   Минимум:  {min(confidence_stats):.1%}")
                print(f"   Максимум: {max(confidence_stats):.1%}")
                print(f"   Среднее:  {np.mean(confidence_stats):.1%}")
                print(f"   Медиана:  {np.median(confidence_stats):.1%}")
                print()
                print(f"   Текущий порог: {confidence_threshold:.1%}")
                print(f"   Минимальная сила: {min_signal_strength} (порог: {strategy.min_strength_threshold:.1%})")
                print()
                
                # Показываем, сколько предсказаний прошло бы пороги
                passed_confidence = sum(1 for c in confidence_stats if c >= confidence_threshold)
                passed_strength = sum(1 for c in confidence_stats if c >= strategy.min_strength_threshold)
                print(f"   Предсказаний с уверенностью >= {confidence_threshold:.1%}: {passed_confidence}/{len(confidence_stats)}")
                print(f"   Предсказаний с уверенностью >= {strategy.min_strength_threshold:.1%}: {passed_strength}/{len(confidence_stats)}")
                print()
            
            if filtered_reasons:
                print(f"🚫 Причины фильтрации сигналов:")
                for reason, count in sorted(filtered_reasons.items(), key=lambda x: x[1], reverse=True):
                    print(f"   {reason:30s}: {count:3d} раз")
                print()
            
            print("💡 Рекомендации:")
            if predictions_stats["HOLD"] / diagnostic_bars > 0.9:
                print("   ⚠️  Модель предсказывает в основном HOLD (>90%)")
                print("      - Возможно, модель слишком консервативна")
                print("      - Рассмотрите возможность переобучения модели")
            if confidence_stats and max(confidence_stats) < confidence_threshold:
                print(f"   ⚠️  Максимальная уверенность ({max(confidence_stats):.1%}) ниже порога ({confidence_threshold:.1%})")
                print(f"      - Попробуйте снизить confidence_threshold до {max(confidence_stats):.1%}")
            if confidence_stats and max(confidence_stats) < strategy.min_strength_threshold:
                print(f"   ⚠️  Максимальная уверенность ({max(confidence_stats):.1%}) ниже порога силы ({strategy.min_strength_threshold:.1%})")
                print(f"      - Попробуйте установить min_signal_strength='слабое'")
            if "сила_слишком_слабая" in filtered_reasons:
                print("   ⚠️  Много сигналов отфильтровано из-за слабой силы")
                print("      - Попробуйте установить min_signal_strength='слабое'")
            if "ожидание_порог" in filtered_reasons:
                print("   ⚠️  Много сигналов отфильтровано из-за порога уверенности")
                print(f"      - Попробуйте снизить confidence_threshold (текущий: {confidence_threshold:.1%})")
            if "индикаторы_не_согласны" in filtered_reasons:
                print("   ⚠️  Много сигналов отфильтровано из-за несогласованности индикаторов")
                print("      - Это нормально, модель проверяет согласованность с RSI/MACD")
            if "объем_не_подтверждает" in filtered_reasons:
                print("   ⚠️  Много сигналов отфильтровано из-за низкого объема")
                print("      - Это нормально, модель проверяет объемное подтверждение")
            print()
            print("🔧 Попробуйте запустить с другими параметрами:")
            print(f"   python test_ml_strategy.py --symbol {symbol} --days {days_back} --confidence 0.5 --strength слабое --no-stability")
            return
        
        # Статистика по типам сигналов
        long_signals = [s for s in signals if s.action == Action.LONG]
        short_signals = [s for s in signals if s.action == Action.SHORT]
        hold_signals = [s for s in signals if s.action == Action.HOLD]
        
        print(f"📊 Распределение сигналов:")
        print(f"   LONG:  {len(long_signals):4d} ({len(long_signals)/len(signals)*100:.1f}%)")
        print(f"   SHORT: {len(short_signals):4d} ({len(short_signals)/len(signals)*100:.1f}%)")
        print(f"   HOLD:  {len(hold_signals):4d} ({len(hold_signals)/len(signals)*100:.1f}%)")
        print()
        
        # Анализ причин HOLD сигналов
        if hold_signals:
            hold_reasons = {}
            for sig in hold_signals:
                reason = sig.reason
                # Группируем похожие причины
                if "insufficient_data" in reason:
                    hold_reasons["Недостаточно данных (<200 баров)"] = hold_reasons.get("Недостаточно данных (<200 баров)", 0) + 1
                elif "нейтрально" in reason or "ожидание" in reason:
                    hold_reasons["Модель предсказывает HOLD"] = hold_reasons.get("Модель предсказывает HOLD", 0) + 1
                elif "сила_слишком_слабая" in reason:
                    hold_reasons["Сила сигнала слишком слабая"] = hold_reasons.get("Сила сигнала слишком слабая", 0) + 1
                elif "порог" in reason or "ожидание_порог" in reason:
                    hold_reasons["Не проходит порог уверенности"] = hold_reasons.get("Не проходит порог уверенности", 0) + 1
                elif "индикаторы_не_согласны" in reason:
                    hold_reasons["Индикаторы не согласны"] = hold_reasons.get("Индикаторы не согласны", 0) + 1
                elif "объем_не_подтверждает" in reason:
                    hold_reasons["Объем не подтверждает"] = hold_reasons.get("Объем не подтверждает", 0) + 1
                elif "слабый_тренд" in reason:
                    hold_reasons["Слабый тренд (ADX < 25)"] = hold_reasons.get("Слабый тренд (ADX < 25)", 0) + 1
                elif "экстремальная_зона" in reason:
                    hold_reasons["Экстремальная зона RSI"] = hold_reasons.get("Экстремальная зона RSI", 0) + 1
                else:
                    hold_reasons[reason[:50]] = hold_reasons.get(reason[:50], 0) + 1
            
            if hold_reasons:
                print("📊 Причины HOLD сигналов:")
                for reason, count in sorted(hold_reasons.items(), key=lambda x: x[1], reverse=True):
                    print(f"   {reason:40s}: {count:4d} ({count/len(hold_signals)*100:.1f}%)")
                print()
        
        # Анализ уверенности
        actionable_signals = [s for s in signals if s.action in (Action.LONG, Action.SHORT)]
        if actionable_signals:
            confidences = []
            for sig in actionable_signals:
                if hasattr(sig, 'indicators_info') and sig.indicators_info:
                    conf = sig.indicators_info.get('confidence', 0)
                    if conf:
                        confidences.append(conf)
            
            if confidences:
                print(f"📊 Статистика уверенности (для {len(confidences)} actionable сигналов):")
                print(f"   Минимум:  {min(confidences):.1%}")
                print(f"   Максимум: {max(confidences):.1%}")
                print(f"   Среднее:  {np.mean(confidences):.1%}")
                print(f"   Медиана:  {np.median(confidences):.1%}")
                print()
        
        # Показываем первые 10 сигналов
        print("=" * 80)
        print("📋 ПЕРВЫЕ 10 СИГНАЛОВ:")
        print("=" * 80)
        for i, sig in enumerate(signals[:10], 1):
            timestamp_str = sig.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(sig.timestamp, 'strftime') else str(sig.timestamp)
            action_str = sig.action.value.upper()
            reason = sig.reason
            
            # Извлекаем уверенность из indicators_info
            confidence_str = ""
            if hasattr(sig, 'indicators_info') and sig.indicators_info:
                conf = sig.indicators_info.get('confidence', 0)
                if conf:
                    confidence_str = f" (confidence: {conf:.1%})"
            
            print(f"{i:2d}. [{timestamp_str}] {action_str:5s} @ ${sig.price:,.2f} - {reason}{confidence_str}")
        
        if len(signals) > 10:
            print(f"\n... и еще {len(signals) - 10} сигналов")
        
        print()
        
        # Показываем последние 10 сигналов
        print("=" * 80)
        print("📋 ПОСЛЕДНИЕ 10 СИГНАЛОВ:")
        print("=" * 80)
        for i, sig in enumerate(signals[-10:], len(signals) - 9):
            timestamp_str = sig.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(sig.timestamp, 'strftime') else str(sig.timestamp)
            action_str = sig.action.value.upper()
            reason = sig.reason
            
            confidence_str = ""
            if hasattr(sig, 'indicators_info') and sig.indicators_info:
                conf = sig.indicators_info.get('confidence', 0)
                if conf:
                    confidence_str = f" (confidence: {conf:.1%})"
            
            print(f"{i:2d}. [{timestamp_str}] {action_str:5s} @ ${sig.price:,.2f} - {reason}{confidence_str}")
        
        print()
        
        # Если мало actionable сигналов, запускаем детальную диагностику
        if len(actionable_signals) < 10 and len(hold_signals) > 200:
            print("=" * 80)
            print("🔍 ДИАГНОСТИКА: Почему все сигналы HOLD?")
            print("=" * 80)
            
            # Загружаем стратегию для диагностики
            from bot.ml.strategy_ml import MLStrategy
            strategy = MLStrategy(model_path, confidence_threshold, min_signal_strength, stability_filter)
            
            # Анализируем последние 200 баров (после начальных 200 для индикаторов)
            diagnostic_start = max(200, len(df_ready) - 200)
            diagnostic_bars = len(df_ready) - diagnostic_start
            
            print(f"📊 Анализируем последние {diagnostic_bars} баров (с {diagnostic_start} по {len(df_ready)})...")
            print()
            
            predictions_stats = {"LONG": 0, "SHORT": 0, "HOLD": 0}
            confidence_stats = []
            filtered_reasons = {}
            
            # Предварительно вычисляем фичи для всего DataFrame
            try:
                df_with_features = strategy.feature_engineer.create_technical_indicators(df_ready)
                print(f"✅ Фичи подготовлены для диагностики")
            except Exception as e:
                print(f"⚠️  Ошибка при подготовке фичей: {e}")
                df_with_features = df_ready
            
            for i in range(diagnostic_start, len(df_ready)):
                try:
                    row = df_with_features.iloc[i] if i < len(df_with_features) else df_ready.iloc[i]
                    df_until_now = df_with_features.iloc[:i+1] if i < len(df_with_features) else df_ready.iloc[:i+1]
                    
                    # Получаем предсказание модели
                    prediction, confidence = strategy.predict(df_until_now, skip_feature_creation=True)
                    
                    if prediction == 1:
                        predictions_stats["LONG"] += 1
                    elif prediction == -1:
                        predictions_stats["SHORT"] += 1
                    else:
                        predictions_stats["HOLD"] += 1
                    
                    if prediction != 0:
                        confidence_stats.append(confidence)
                    
                    # Пробуем сгенерировать сигнал для диагностики
                    try:
                        from bot.strategy import Bias
                        test_signal = strategy.generate_signal(
                            row=row,
                            df=df_until_now,
                            has_position=None,
                            current_price=row["close"],
                            leverage=10,
                            target_profit_pct_margin=25.0,
                            max_loss_pct_margin=10.0,
                        )
                        if test_signal.action == Action.HOLD and test_signal.reason.startswith("ml_"):
                            # Извлекаем причину фильтрации
                            reason = test_signal.reason
                            if "сила_слишком_слабая" in reason:
                                filtered_reasons["сила_слишком_слабая"] = filtered_reasons.get("сила_слишком_слабая", 0) + 1
                            elif "ожидание" in reason or "порог" in reason:
                                filtered_reasons["не_проходит_порог"] = filtered_reasons.get("не_проходит_порог", 0) + 1
                            elif "индикаторы_не_согласны" in reason:
                                filtered_reasons["индикаторы_не_согласны"] = filtered_reasons.get("индикаторы_не_согласны", 0) + 1
                            elif "объем_не_подтверждает" in reason:
                                filtered_reasons["объем_не_подтверждает"] = filtered_reasons.get("объем_не_подтверждает", 0) + 1
                            elif "слабый_тренд" in reason:
                                filtered_reasons["слабый_тренд"] = filtered_reasons.get("слабый_тренд", 0) + 1
                            elif "экстремальная_зона" in reason:
                                filtered_reasons["экстремальная_зона"] = filtered_reasons.get("экстремальная_зона", 0) + 1
                            elif "нейтрально" in reason:
                                filtered_reasons["модель_предсказывает_HOLD"] = filtered_reasons.get("модель_предсказывает_HOLD", 0) + 1
                    except:
                        pass
                except Exception as e:
                    continue
            
            print(f"📊 Статистика предсказаний модели (последние {diagnostic_bars} баров):")
            print(f"   LONG:  {predictions_stats['LONG']:3d} ({predictions_stats['LONG']/diagnostic_bars*100:.1f}%)")
            print(f"   SHORT: {predictions_stats['SHORT']:3d} ({predictions_stats['SHORT']/diagnostic_bars*100:.1f}%)")
            print(f"   HOLD:  {predictions_stats['HOLD']:3d} ({predictions_stats['HOLD']/diagnostic_bars*100:.1f}%)")
            print()
            
            if confidence_stats:
                print(f"📊 Статистика уверенности (для {len(confidence_stats)} actionable предсказаний):")
                print(f"   Минимум:  {min(confidence_stats):.1%}")
                print(f"   Максимум: {max(confidence_stats):.1%}")
                print(f"   Среднее:  {np.mean(confidence_stats):.1%}")
                print(f"   Медиана:  {np.median(confidence_stats):.1%}")
                print()
                print(f"   Текущий порог: {confidence_threshold:.1%}")
                print(f"   Минимальная сила: {min_signal_strength} (порог: {strategy.min_strength_threshold:.1%})")
                print()
                
                # Показываем, сколько предсказаний прошло бы пороги
                passed_confidence = sum(1 for c in confidence_stats if c >= confidence_threshold)
                passed_strength = sum(1 for c in confidence_stats if c >= strategy.min_strength_threshold)
                print(f"   Предсказаний с уверенностью >= {confidence_threshold:.1%}: {passed_confidence}/{len(confidence_stats)}")
                print(f"   Предсказаний с уверенностью >= {strategy.min_strength_threshold:.1%}: {passed_strength}/{len(confidence_stats)}")
                print()
            
            if filtered_reasons:
                print(f"🚫 Причины фильтрации сигналов:")
                for reason, count in sorted(filtered_reasons.items(), key=lambda x: x[1], reverse=True):
                    print(f"   {reason:30s}: {count:3d} раз")
                print()
            
            print("💡 Рекомендации:")
            if predictions_stats["HOLD"] / diagnostic_bars > 0.9:
                print("   ⚠️  Модель предсказывает в основном HOLD (>90%)")
                print("      - Возможно, модель слишком консервативна")
                print("      - Рассмотрите возможность переобучения модели")
            if confidence_stats and max(confidence_stats) < confidence_threshold:
                print(f"   ⚠️  Максимальная уверенность ({max(confidence_stats):.1%}) ниже порога ({confidence_threshold:.1%})")
                print(f"      - Попробуйте снизить confidence_threshold до {max(confidence_stats):.1%}")
            if confidence_stats and max(confidence_stats) < strategy.min_strength_threshold:
                print(f"   ⚠️  Максимальная уверенность ({max(confidence_stats):.1%}) ниже порога силы ({strategy.min_strength_threshold:.1%})")
                print(f"      - Попробуйте установить min_signal_strength='слабое'")
            if "сила_слишком_слабая" in filtered_reasons:
                print("   ⚠️  Много сигналов отфильтровано из-за слабой силы")
                print("      - Попробуйте установить min_signal_strength='слабое'")
            if "не_проходит_порог" in filtered_reasons:
                print("   ⚠️  Много сигналов отфильтровано из-за порога уверенности")
                print(f"      - Попробуйте снизить confidence_threshold (текущий: {confidence_threshold:.1%})")
            if "индикаторы_не_согласны" in filtered_reasons:
                print("   ⚠️  Много сигналов отфильтровано из-за несогласованности индикаторов")
                print("      - Это нормально, модель проверяет согласованность с RSI/MACD")
            if "объем_не_подтверждает" in filtered_reasons:
                print("   ⚠️  Много сигналов отфильтровано из-за низкого объема")
                print("      - Это нормально, модель проверяет объемное подтверждение")
            print()
        
        # Анализ причин (из reason) для actionable сигналов
        if actionable_signals:
            print("=" * 80)
            print("🔍 АНАЛИЗ ПРИЧИН ACTIONABLE СИГНАЛОВ:")
            print("=" * 80)
            reason_counts = {}
            for sig in actionable_signals:
                reason = sig.reason
                # Извлекаем основную причину (до первого подчеркивания после ml_)
                if reason.startswith("ml_"):
                    parts = reason.split("_")
                    if len(parts) >= 2:
                        main_reason = f"{parts[0]}_{parts[1]}"
                        reason_counts[main_reason] = reason_counts.get(main_reason, 0) + 1
            
            for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"   {reason:30s}: {count:4d} сигналов")
            print()
        
        print("=" * 80)
        print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ ОШИБКА при генерации сигналов: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Тестирование ML стратегии на исторических данных")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Торговая пара (по умолчанию: BTCUSDT)")
    parser.add_argument("--days", type=int, default=7, help="Количество дней назад для тестирования (по умолчанию: 7)")
    parser.add_argument("--model", type=str, default=None, help="Путь к модели (по умолчанию: авто-поиск)")
    parser.add_argument("--confidence", type=float, default=None, help="Порог уверенности (по умолчанию: из настроек)")
    parser.add_argument("--strength", type=str, default=None, help="Минимальная сила сигнала (по умолчанию: из настроек)")
    parser.add_argument("--no-stability", action="store_true", help="Отключить фильтр стабильности")
    
    args = parser.parse_args()
    
    # Загружаем настройки для получения значений по умолчанию
    settings = load_settings()
    
    confidence = args.confidence if args.confidence is not None else settings.ml_confidence_threshold
    strength = args.strength if args.strength is not None else settings.ml_min_signal_strength
    stability = not args.no_stability
    
    test_ml_strategy(
        symbol=args.symbol,
        days_back=args.days,
        model_path=args.model,
        confidence_threshold=confidence,
        min_signal_strength=strength,
        stability_filter=stability,
    )
