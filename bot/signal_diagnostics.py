"""
signal_diagnostics.py - ОБНОВЛЕННАЯ ВЕРСИЯ
С обработкой пропущенных значений индикаторов
"""
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
import json
from pathlib import Path


class SignalDiagnostics:
    """
    Класс для диагностики и мониторинщения торговых сигналов
    С корректной обработкой пропущенных значений индикаторов
    """
    
    # Константы
    SIGNAL_FRESHNESS_SECONDS = 60  # Сигнал считается свежим (сек)
    DIAGNOSTIC_INTERVAL = 300  # Интервал диагностики (сек)
    MIN_VALID_DATA_RATIO = 0.7  # Минимальный процент валидных данных
    INDICATOR_WARMUP_PERIODS = {
        'adx': 28,  # ADX обычно требует 14*2 периодов
        'atr': 14,
        'rsi': 14,
        'sma': 20,
        'ema_fast': 12,
        'ema_slow': 26,
        'bb_upper': 20,
        'bb_lower': 20,
        'atr_1h': 14,
        'atr_4h': 14,
        'atr_avg': 14,
        'ema_fast_1h': 12,
        'ema_slow_1h': 26,
        'vol_sma': 20,
    }
    
    def __init__(self, symbol: str, log_func=None):
        self.symbol = symbol
        self.log_func = log_func if log_func else self._default_log
        self.last_diagnostic_time = None
        self.diagnostic_history = []
        self.max_history_size = 100
        
    def _default_log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{self.symbol}] [{level}] {message}")
    
    def check_signal_generation(
        self,
        all_signals: List[Any],
        strategy_settings: Any,
        df_ready: pd.DataFrame,
        current_price: float
    ) -> Dict[str, Any]:
        """
        Основная функция диагностики с улучшенной обработкой данных
        """
        # Проверяем, нужно ли выполнять диагностику
        current_time = datetime.now(timezone.utc)
        if (self.last_diagnostic_time and 
            (current_time - self.last_diagnostic_time).total_seconds() < self.DIAGNOSTIC_INTERVAL):
            return {"diagnostics_performed": False}
        
        self.last_diagnostic_time = current_time
        
        # Выполняем полную диагностику с улучшенной обработкой данных
        diagnostic_result = {
            "diagnostics_performed": True,
            "timestamp": current_time.isoformat(),
            "symbol": self.symbol,
            "current_price": current_price,
            "market_conditions": self._analyze_market_conditions_improved(df_ready),
            "strategies_analysis": self._analyze_strategies(strategy_settings),
            "signal_analysis": self._analyze_signals(all_signals, current_time),
            "data_quality": self._analyze_data_quality_improved(df_ready),
            "recommendations": []
        }
        
        # Генерируем рекомендации
        self._generate_recommendations_improved(diagnostic_result)
        
        # Сохраняем в историю
        self._save_to_history(diagnostic_result)
        
        # Логируем результаты
        self._log_diagnostics_improved(diagnostic_result)
        
        return diagnostic_result
    
    def _analyze_market_conditions_improved(self, df_ready: pd.DataFrame) -> Dict[str, Any]:
        """Улучшенный анализ рыночных условий с обработкой пропущенных значений"""
        if df_ready.empty:
            return {"error": "Empty dataframe", "has_sufficient_data": False}
        
        market_conditions = {
            "has_sufficient_data": False,
            "total_rows": len(df_ready),
            "price": 0.0,
            "volume": 0.0,
            "market_phase": "unknown",
            "market_bias": "unknown",
            "indicators": {},
            "data_warnings": []
        }
        
        # Проверяем достаточно ли данных
        if len(df_ready) < 50:
            market_conditions["data_warnings"].append(
                f"Мало данных: {len(df_ready)} строк (рекомендуется ≥50)"
            )
            return market_conditions
        
        # Получаем последнюю строку с учетом того, что индикаторы могут быть NaN
        last_row = df_ready.iloc[-1]
        
        # Базовая информация (всегда должна быть)
        market_conditions["price"] = float(last_row.get("close", 0))
        market_conditions["volume"] = float(last_row.get("volume", 0))
        
        # Анализ последних N валидных строк для каждого индикатора
        valid_data_window = 20  # Анализируем последние 20 валидных значений
        
        # Ключевые индикаторы для проверки
        key_indicators = ['adx', 'rsi', 'atr', 'sma', 'ema_fast', 'ema_slow', 
                         'bb_upper', 'bb_lower', 'vol_sma']
        
        indicator_status = {}
        
        for indicator in key_indicators:
            if indicator in df_ready.columns:
                # Получаем последние N значений
                recent_values = df_ready[indicator].dropna().tail(valid_data_window)
                
                if len(recent_values) > 0:
                    # Последнее значение
                    last_value = float(recent_values.iloc[-1])
                    market_conditions["indicators"][indicator] = last_value
                    
                    # Процент валидных данных
                    total_rows = min(valid_data_window, len(df_ready))
                    valid_pct = (len(recent_values) / total_rows) * 100
                    
                    indicator_status[indicator] = {
                        "has_data": True,
                        "value": last_value,
                        "valid_pct": round(valid_pct, 1),
                        "warmup_complete": valid_pct >= 80  # 80% валидных данных
                    }
                else:
                    indicator_status[indicator] = {
                        "has_data": False,
                        "warmup_complete": False,
                        "message": f"Нет валидных значений для {indicator}"
                    }
            else:
                indicator_status[indicator] = {
                    "has_data": False,
                    "warmup_complete": False,
                    "message": f"Индикатор {indicator} отсутствует в данных"
                }
        
        market_conditions["indicator_status"] = indicator_status
        
        # Определяем, достаточно ли данных для анализа
        key_indicators_for_analysis = ['adx', 'rsi', 'atr']
        indicators_with_data = [ind for ind in key_indicators_for_analysis 
                              if indicator_status.get(ind, {}).get('has_data', False)]
        
        if len(indicators_with_data) >= 2:
            market_conditions["has_sufficient_data"] = True
            
            # Определяем фазу рынка (только если есть достаточные данные)
            try:
                from bot.strategy import detect_market_phase, detect_market_bias
                
                # Создаем "очищенную" последнюю строку с заполненными NaN
                clean_last_row = last_row.copy()
                
                # Заполняем пропущенные значения ближайшими валидными
                for col in df_ready.columns:
                    if pd.isna(clean_last_row[col]):
                        valid_values = df_ready[col].dropna()
                        if len(valid_values) > 0:
                            clean_last_row[col] = valid_values.iloc[-1]
                
                # Пытаемся определить фазу
                phase = detect_market_phase(clean_last_row, None)
                if phase:
                    market_conditions["market_phase"] = phase.value
                else:
                    # Fallback по ADX
                    adx = market_conditions["indicators"].get("adx")
                    if adx is not None:
                        market_conditions["market_phase"] = "trend" if adx > 25 else "flat"
                
                # Пытаемся определить bias
                bias = detect_market_bias(clean_last_row)
                if bias:
                    market_conditions["market_bias"] = bias.value
                else:
                    # Fallback по цене относительно SMA/EMA
                    price = market_conditions["price"]
                    sma = market_conditions["indicators"].get("sma")
                    ema_fast = market_conditions["indicators"].get("ema_fast")
                    
                    if sma is not None:
                        market_conditions["market_bias"] = "long" if price > sma else "short"
                    elif ema_fast is not None:
                        market_conditions["market_bias"] = "long" if price > ema_fast else "short"
                        
            except Exception as e:
                market_conditions["data_warnings"].append(f"Ошибка анализа рынка: {str(e)}")
        
        # Анализ объема
        if "vol_sma" in df_ready.columns:
            vol_sma_status = indicator_status.get("vol_sma", {})
            if vol_sma_status.get("has_data"):
                volume = market_conditions["volume"]
                vol_sma = vol_sma_status["value"]
                if vol_sma > 0:
                    market_conditions["volume_ratio"] = volume / vol_sma
                    market_conditions["volume_status"] = (
                        "high" if volume > vol_sma * 1.5 else
                        "low" if volume < vol_sma * 0.5 else "normal"
                    )
        
        return market_conditions
    
    def _analyze_data_quality_improved(self, df_ready: pd.DataFrame) -> Dict[str, Any]:
        """Улучшенный анализ качества данных"""
        if df_ready.empty:
            return {
                "quality": "poor",
                "issues": ["Dataframe is empty"],
                "recommendation": "Увеличьте kline_limit в настройках"
            }
        
        total_rows = len(df_ready)
        data_quality = {
            "total_rows": total_rows,
            "quality": "good",
            "issues": [],
            "warnings": [],
            "missing_data_summary": {},
            "recommendations": []
        }
        
        # Ключевые колонки, которые должны быть
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df_ready.columns:
                data_quality["issues"].append(f"Отсутствует обязательная колонка: {col}")
                data_quality["quality"] = "poor"
        
        if data_quality["quality"] == "poor":
            return data_quality
        
        # Анализ пропущенных значений для индикаторов
        indicator_columns = [col for col in df_ready.columns if col not in required_columns]
        
        missing_summary = {}
        problem_indicators = []
        
        for column in indicator_columns:
            null_count = df_ready[column].isnull().sum()
            if null_count > 0:
                null_pct = (null_count / total_rows) * 100
                
                missing_summary[column] = {
                    "null_count": int(null_count),
                    "null_percentage": round(null_pct, 1),
                    "valid_count": total_rows - null_count,
                    "valid_percentage": round(100 - null_pct, 1)
                }
                
                # Определяем, является ли это проблемой
                warmup_period = self.INDICATOR_WARMUP_PERIODS.get(column, 20)
                expected_nulls = min(warmup_period, total_rows)
                
                if null_count > expected_nulls * 1.5:  # На 50% больше ожидаемого
                    problem_indicators.append({
                        "column": column,
                        "null_count": null_count,
                        "null_pct": round(null_pct, 1),
                        "expected_nulls": expected_nulls,
                        "severity": "high" if null_pct > 50 else "medium"
                    })
                    data_quality["warnings"].append(
                        f"Индикатор '{column}' имеет {null_count} пропусков ({null_pct:.1f}%)"
                    )
        
        data_quality["missing_data_summary"] = missing_summary
        
        # Анализ проблемы с ADX и ATR
        for indicator in ['adx', 'atr_4h', 'atr_avg', 'ema_slow_1h']:
            if indicator in missing_summary:
                info = missing_summary[indicator]
                if info["null_percentage"] > 30:
                    data_quality["recommendations"].append(
                        f"Индикатор {indicator}: {info['null_count']} пропусков. "
                        f"Это нормально для начала расчета индикатора. "
                        f"Валидных значений: {info['valid_count']} ({info['valid_percentage']}%)"
                    )
        
        # Проверка свежести данных
        try:
            if hasattr(df_ready.index, '__getitem__'):
                last_timestamp = df_ready.index[-1]
                if isinstance(last_timestamp, pd.Timestamp):
                    last_time = last_timestamp.to_pydatetime()
                    if last_time.tzinfo is None:
                        last_time = last_time.replace(tzinfo=timezone.utc)
                    else:
                        last_time = last_time.astimezone(timezone.utc)
                    
                    current_time = datetime.now(timezone.utc)
                    age_seconds = (current_time - last_time).total_seconds()
                    
                    data_quality["data_freshness"] = {
                        "last_data_time": last_time.isoformat(),
                        "age_seconds": age_seconds,
                        "status": "fresh" if age_seconds < 300 else "stale"
                    }
                    
                    if age_seconds > 600:  # 10 минут
                        data_quality["issues"].append(f"Данные устарели: {age_seconds:.0f} секунд")
                        data_quality["quality"] = "poor"
        except Exception as e:
            data_quality["warnings"].append(f"Не удалось проверить свежесть данных: {str(e)}")
        
        # Общая оценка качества
        if len(data_quality["issues"]) > 0:
            data_quality["quality"] = "poor"
        elif len(data_quality["warnings"]) > 3:
            data_quality["quality"] = "warning"
        
        return data_quality
    
    def _generate_recommendations_improved(self, result: Dict[str, Any]) -> None:
        """Улучшенная генерация рекомендаций с учетом качества данных"""
        recommendations = []
        
        # Анализ сигналов
        signal_analysis = result["signal_analysis"]
        
        if signal_analysis["total"] == 0:
            recommendations.append("❌ Нет сигналов от стратегий.")
        
        # Проверка качества данных
        data_quality = result.get("data_quality", {})
        if data_quality.get("quality") == "poor":
            recommendations.append("⚠️ Проблемы с качеством данных:")
            for issue in data_quality.get("issues", [])[:3]:  # Показываем первые 3 проблемы
                recommendations.append(f"   - {issue}")
        
        # Проверка достаточности данных для анализа рынка
        market_conditions = result["market_conditions"]
        if not market_conditions.get("has_sufficient_data", False):
            recommendations.append("📊 Недостаточно данных для анализа рынка:")
            
            # Проверяем конкретные индикаторы
            indicator_status = market_conditions.get("indicator_status", {})
            for indicator, status in indicator_status.items():
                if not status.get("has_data", False):
                    recommendations.append(f"   - {indicator}: {status.get('message', 'нет данных')}")
            
            # Общая рекомендация
            recommendations.append("   Рекомендация: Подождите несколько минут пока накопятся данные")
        
        # Проверка пропущенных значений (нормализация сообщений)
        missing_summary = data_quality.get("missing_data_summary", {})
        high_missing_indicators = []
        
        for indicator, info in missing_summary.items():
            if info.get("null_percentage", 0) > 40:  # Более 40% пропусков
                high_missing_indicators.append(
                    f"{indicator}: {info['valid_count']}/{info['valid_count'] + info['null_count']} "
                    f"({info['valid_percentage']}%) валидных"
                )
        
        if high_missing_indicators and len(high_missing_indicators) > 0:
            recommendations.append("📈 Индикаторы в процессе расчета:")
            for indicator_info in high_missing_indicators[:3]:  # Показываем первые 3
                recommendations.append(f"   - {indicator_info}")
            recommendations.append("   Это нормально в начале работы или после перезапуска")
        
        # Анализ стратегий
        strategies = result["strategies_analysis"]
        active_strategies = [name for name, info in strategies.items() if info["enabled"]]
        
        if not active_strategies:
            recommendations.append("❌ Все стратегии отключены.")
        else:
            # Проверяем соответствие стратегий рыночным условиям
            market_phase = market_conditions.get("market_phase")
            
            if market_phase == "flat" and not strategies.get("FLAT", {}).get("enabled"):
                recommendations.append("💡 Рынок в FLAT фазе - включите FLAT стратегию")
            
            if market_phase == "trend" and not strategies.get("TREND", {}).get("enabled"):
                recommendations.append("💡 Рынок в TREND фазе - включите TREND стратегию")
            
            if market_phase == "unknown":
                recommendations.append("🔍 Фаза рынка не определена - проверьте расчет индикаторов")
        
        # Проверка свежести сигналов
        if signal_analysis["fresh_signals"] == 0 and signal_analysis["total"] > 0:
            recommendations.append("⚠️ Есть сигналы, но все устарели (>60с)")
        
        result["recommendations"] = recommendations
    
    def _log_diagnostics_improved(self, result: Dict[str, Any]) -> None:
        """Улучшенное логирование диагностики"""
        self.log_func("=" * 70)
        self.log_func("📊 ДИАГНОСТИКА СИГНАЛОВ (УЛУЧШЕННАЯ)")
        self.log_func(f"Время: {result['timestamp']}")
        self.log_func(f"Символ: {result['symbol']}")
        self.log_func(f"Цена: ${result['current_price']:.2f}")
        self.log_func("-" * 70)
        
        # Качество данных
        data_quality = result.get("data_quality", {})
        quality_icon = "✅" if data_quality.get("quality") == "good" else "⚠️" if data_quality.get("quality") == "warning" else "❌"
        self.log_func(f"{quality_icon} КАЧЕСТВО ДАННЫХ: {data_quality.get('quality', 'unknown').upper()}")
        self.log_func(f"  Всего строк: {data_quality.get('total_rows', 0)}")
        
        # Анализ пропущенных значений
        missing_summary = data_quality.get("missing_data_summary", {})
        if missing_summary:
            self.log_func("  ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ (нормально в начале работы):")
            for indicator, info in list(missing_summary.items())[:5]:  # Показываем первые 5
                self.log_func(f"    {indicator}: {info['valid_count']} валидных, {info['null_count']} пропусков ({info['valid_percentage']}% валидных)")
        
        # Рыночные условия
        market = result["market_conditions"]
        self.log_func("📈 РЫНОЧНЫЕ УСЛОВИЯ:")
        
        if not market.get("has_sufficient_data", False):
            self.log_func("  ⚠️ НЕДОСТАТОЧНО ДАННЫХ для анализа")
            for warning in market.get("data_warnings", [])[:3]:
                self.log_func(f"    {warning}")
        else:
            self.log_func(f"  Фаза: {market.get('market_phase', 'unknown')}")
            self.log_func(f"  Направление: {market.get('market_bias', 'unknown')}")
            self.log_func(f"  Цена: ${market.get('price', 0):.2f}")
            
            # Ключевые индикаторы
            indicators = market.get("indicators", {})
            if indicators:
                self.log_func("  Ключевые индикаторы:")
                for ind in ['adx', 'rsi', 'atr', 'sma']:
                    if ind in indicators:
                        self.log_func(f"    {ind.upper()}: {indicators[ind]:.2f}")
        
        # Статус стратегий
        self.log_func("⚙️ СТРАТЕГИИ:")
        strategies = result["strategies_analysis"]
        active_count = sum(1 for info in strategies.values() if info["enabled"])
        self.log_func(f"  Активных: {active_count}/{len(strategies)}")
        
        # Показываем только активные стратегии
        active_strategies = [name for name, info in strategies.items() if info["enabled"]]
        if active_strategies:
            self.log_func(f"  ✅ Активные: {', '.join(active_strategies)}")
        else:
            self.log_func("  ❌ Нет активных стратегий!")
        
        # Сигналы
        signals = result["signal_analysis"]
        self.log_func("📡 СИГНАЛЫ:")
        self.log_func(f"  Всего: {signals['total']}")
        self.log_func(f"  Свежих (≤{self.SIGNAL_FRESHNESS_SECONDS}с): {signals['fresh_signals']}")
        
        if signals['total'] > 0:
            # Детали по действиям
            self.log_func("  По действиям:")
            for action, count in signals['by_action'].items():
                if count > 0:
                    self.log_func(f"    {action}: {count}")
            
            # Свежие сигналы
            fresh_signals = [s for s in signals['signal_details'] if s.get('is_fresh')]
            if fresh_signals:
                self.log_func("  🎯 СВЕЖИЕ СИГНАЛЫ:")
                for sig in fresh_signals[:3]:
                    self.log_func(f"    {sig['action']} @ ${sig['price']:.2f} ({sig['strategy']}) - {sig['reason'][:50]}...")
        
        # Рекомендации
        if result["recommendations"]:
            self.log_func("💡 РЕКОМЕНДАЦИИ:")
            for rec in result["recommendations"]:
                self.log_func(f"  {rec}")
        
        self.log_func("=" * 70)

    # Остальные методы остаются без изменений (но могут использовать улучшенные версии)
    def _analyze_strategies(self, strategy_settings: Any) -> Dict[str, Any]:
        """Анализ настроек стратегий"""
        strategies_analysis = {}
        
        strategy_configs = [
            ("TREND", "enable_trend_strategy"),
            ("FLAT", "enable_flat_strategy"),
            ("ML", "enable_ml_strategy"),
            ("MOMENTUM", "enable_momentum_strategy"),
            ("LIQUIDITY", "enable_liquidity_sweep_strategy"),
            ("SMC", "enable_smc_strategy"),
            ("ICT", "enable_ict_strategy"),
            ("ZSCORE", "enable_zscore_strategy"),
            ("VBO", "enable_vbo_strategy"),
            ("AMT_OF", "enable_amt_of_strategy"),
        ]
        
        for display_name, attr_name in strategy_configs:
            enabled = False
            reason = "Setting not found"
            
            try:
                if hasattr(strategy_settings, attr_name):
                    enabled = getattr(strategy_settings, attr_name)
                    reason = "ENABLED" if enabled else "DISABLED"
                elif isinstance(strategy_settings, dict):
                    enabled = strategy_settings.get(attr_name, False)
                    reason = "ENABLED" if enabled else "DISABLED"
                else:
                    reason = "Cannot access settings"
            except Exception as e:
                reason = f"Error: {str(e)}"
            
            strategies_analysis[display_name] = {
                "enabled": enabled,
                "status": "ACTIVE" if enabled else "INACTIVE",
                "reason": reason
            }
        
        return strategies_analysis
    
    def _analyze_signals(self, all_signals: List[Any], current_time: datetime) -> Dict[str, Any]:
        """Анализ сгенерированных сигналов"""
        signal_analysis = {
            "total": len(all_signals),
            "by_action": {"LONG": 0, "SHORT": 0, "HOLD": 0},
            "by_strategy": {},
            "fresh_signals": 0,
            "signal_details": [],
            "problems": []
        }
        
        for signal in all_signals:
            try:
                # Определяем действие
                if hasattr(signal, 'action'):
                    action = signal.action
                    if hasattr(action, 'value'):
                        action_name = action.value
                    else:
                        action_name = str(action)
                else:
                    action_name = "UNKNOWN"
                
                signal_analysis["by_action"][action_name] = (
                    signal_analysis["by_action"].get(action_name, 0) + 1
                )
                
                # Определяем стратегию
                if hasattr(signal, 'reason'):
                    reason = signal.reason
                    strategy_type = self._get_strategy_type(reason)
                else:
                    reason = "unknown"
                    strategy_type = "unknown"
                
                signal_analysis["by_strategy"][strategy_type] = (
                    signal_analysis["by_strategy"].get(strategy_type, 0) + 1
                )
                
                # Проверяем свежесть
                is_fresh = False
                signal_time = None
                age_seconds = 9999
                
                if hasattr(signal, 'timestamp'):
                    signal_time = signal.timestamp
                    
                    # Конвертируем в datetime
                    if isinstance(signal_time, pd.Timestamp):
                        if signal_time.tzinfo is None:
                            signal_time = signal_time.tz_localize('UTC')
                        else:
                            signal_time = signal_time.tz_convert('UTC')
                        signal_dt = signal_time.to_pydatetime()
                    elif isinstance(signal_time, datetime):
                        if signal_time.tzinfo is None:
                            signal_dt = signal_time.replace(tzinfo=timezone.utc)
                        else:
                            signal_dt = signal_time
                    else:
                        signal_dt = current_time
                    
                    age_seconds = (current_time - signal_dt).total_seconds()
                    is_fresh = age_seconds <= self.SIGNAL_FRESHNESS_SECONDS
                
                if is_fresh:
                    signal_analysis["fresh_signals"] += 1
                
                # Сохраняем детали
                signal_analysis["signal_details"].append({
                    "action": action_name,
                    "strategy": strategy_type,
                    "reason": reason,
                    "price": getattr(signal, 'price', 0) if hasattr(signal, 'price') else 0,
                    "timestamp": signal_dt.isoformat() if signal_dt else None,
                    "age_seconds": age_seconds,
                    "is_fresh": is_fresh
                })
                
            except Exception as e:
                signal_analysis["problems"].append(f"Error analyzing signal: {e}")
        
        return signal_analysis
    
    def _get_strategy_type(self, reason: str) -> str:
        """Определение типа стратегии по reason"""
        reason_lower = reason.lower()
        
        strategy_mapping = [
            ("ml_", "ml"),
            ("trend_", "trend"),
            ("range_", "flat"),
            ("momentum_", "momentum"),
            ("liquidity_", "liquidity"),
            ("smc_", "smc"),
            ("ict_", "ict"),
            ("zscore_", "zscore"),
            ("vbo_", "vbo"),
            ("amt_of_", "amt_of"),
        ]
        
        for prefix, strategy_type in strategy_mapping:
            if reason_lower.startswith(prefix):
                return strategy_type
        
        return "unknown"
    
    def _save_to_history(self, diagnostic_result: Dict[str, Any]) -> None:
        """Сохранение диагностики в историю"""
        simplified = {
            "timestamp": diagnostic_result["timestamp"],
            "symbol": diagnostic_result["symbol"],
            "total_signals": diagnostic_result["signal_analysis"]["total"],
            "fresh_signals": diagnostic_result["signal_analysis"]["fresh_signals"],
            "market_phase": diagnostic_result["market_conditions"].get("market_phase"),
            "price": diagnostic_result["current_price"],
            "recommendations": diagnostic_result["recommendations"],
            "data_quality": diagnostic_result.get("data_quality", {}).get("quality", "unknown")
        }
        
        self.diagnostic_history.append(simplified)
        
        if len(self.diagnostic_history) > self.max_history_size:
            self.diagnostic_history = self.diagnostic_history[-self.max_history_size:]
    
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Получение истории диагностики"""
        return self.diagnostic_history[-limit:]
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Получение статистики диагностики"""
        if not self.diagnostic_history:
            return {"total_diagnostics": 0}
        
        total = len(self.diagnostic_history)
        avg_signals = sum(d.get("total_signals", 0) for d in self.diagnostic_history) / total
        avg_fresh = sum(d.get("fresh_signals", 0) for d in self.diagnostic_history) / total
        
        # Анализ качества данных в истории
        data_quality_counts = {}
        for d in self.diagnostic_history:
            quality = d.get("data_quality", "unknown")
            data_quality_counts[quality] = data_quality_counts.get(quality, 0) + 1
        
        return {
            "total_diagnostics": total,
            "avg_signals_per_check": round(avg_signals, 1),
            "avg_fresh_signals_per_check": round(avg_fresh, 1),
            "data_quality_summary": data_quality_counts,
            "last_check": self.diagnostic_history[-1]["timestamp"] if self.diagnostic_history else None
        }


# Функция для быстрой проверки пропущенных значений
def check_missing_values_report(df: pd.DataFrame, symbol: str) -> str:
    """
    Быстрый отчет о пропущенных значениях в индикаторах
    
    Returns:
        Строка с отчетом
    """
    if df.empty:
        return f"[{symbol}] ❌ DataFrame пустой"
    
    report_lines = []
    report_lines.append(f"\n{'='*60}")
    report_lines.append(f"📊 ОТЧЕТ О ПРОПУЩЕННЫХ ЗНАЧЕНИЯХ: {symbol}")
    report_lines.append(f"{'='*60}")
    report_lines.append(f"Всего строк: {len(df)}")
    report_lines.append(f"Колонок: {len(df.columns)}")
    report_lines.append("-" * 60)
    
    # Обязательные колонки
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_required = [col for col in required_cols if col not in df.columns]
    
    if missing_required:
        report_lines.append("❌ ОТСУТСТВУЮТ ОБЯЗАТЕЛЬНЫЕ КОЛОНКИ:")
        for col in missing_required:
            report_lines.append(f"  - {col}")
    else:
        report_lines.append("✅ Все обязательные колонки присутствуют")
    
    # Анализ индикаторов
    indicator_cols = [col for col in df.columns if col not in required_cols]
    
    if indicator_cols:
        report_lines.append("\n📈 СТАТУС ИНДИКАТОРОВ:")
        
        # Группируем по типу
        indicator_groups = {
            "Трендовые": ['adx', 'plus_di', 'minus_di', 'sma', 'ema_fast', 'ema_slow'],
            "Волатильность": ['atr', 'atr_1h', 'atr_4h', 'atr_avg', 'bb_upper', 'bb_lower'],
            "Осцилляторы": ['rsi'],
            "Объем": ['vol_sma'],
            "Другие": []
        }
        
        for group_name, group_indicators in indicator_groups.items():
            group_report = []
            for indicator in group_indicators:
                if indicator in df.columns:
                    null_count = df[indicator].isnull().sum()
                    total = len(df)
                    if null_count > 0:
                        valid_pct = ((total - null_count) / total) * 100
                        status = "✅" if valid_pct > 80 else "⚠️" if valid_pct > 50 else "❌"
                        group_report.append(f"{status} {indicator}: {total - null_count}/{total} ({valid_pct:.1f}%) валидных")
                    else:
                        group_report.append(f"✅ {indicator}: 100% валидных")
                elif indicator in [i for g in indicator_groups.values() for i in g]:
                    group_report.append(f"❌ {indicator}: отсутствует")
            
            if group_report:
                report_lines.append(f"\n{group_name}:")
                for line in group_report:
                    report_lines.append(f"  {line}")
        
        # Анализ свежести последних данных
        report_lines.append("\n⏱️ СВЕЖЕСТЬ ДАННЫХ:")
        
        # Проверяем несколько ключевых индикаторов
        key_indicators = ['adx', 'rsi', 'atr', 'sma']
        for indicator in key_indicators:
            if indicator in df.columns:
                valid_values = df[indicator].dropna()
                if len(valid_values) > 0:
                    last_valid_idx = df[indicator].last_valid_index()
                    if last_valid_idx is not None:
                        report_lines.append(f"  {indicator}: последнее значение на строке {df.index.get_loc(last_valid_idx)}")
                    else:
                        report_lines.append(f"  {indicator}: нет валидных значений")
                else:
                    report_lines.append(f"  {indicator}: нет валидных значений")
    
    report_lines.append(f"\n{'='*60}")
    report_lines.append("💡 РЕКОМЕНДАЦИИ:")
    
    if len(df) < 100:
        report_lines.append("1. Увеличьте kline_limit в настройках до 200+")
    
    # Считаем индикаторы с недостаточными данными
    indicators_with_issues = []
    for col in indicator_cols:
        if col in df.columns:
            valid_pct = (df[col].notna().sum() / len(df)) * 100
            if valid_pct < 70:
                indicators_with_issues.append(f"{col} ({valid_pct:.1f}% валидных)")
    
    if indicators_with_issues:
        report_lines.append("2. Следующие индикаторы в процессе расчета:")
        for issue in indicators_with_issues[:5]:  # Показываем первые 5
            report_lines.append(f"   - {issue}")
        report_lines.append("   Это нормально в начале работы бота")
    
    report_lines.append(f"{'='*60}\n")
    
    return "\n".join(report_lines)


def quick_signal_check(symbol: str, all_signals: List[Any], strategies_enabled: Dict[str, bool]) -> Dict[str, Any]:
    """
    Быстрая проверка наличия и свежести сигналов.
    
    Args:
        symbol: Символ
        all_signals: Список всех сигналов
        strategies_enabled: Словарь включенных стратегий
        
    Returns:
        Словарь с краткой статистикой
    """
    fresh_threshold = 60  # 1 минута
    current_time = datetime.now(timezone.utc)
    
    fresh_signals = 0
    total_signals = len(all_signals)
    
    for sig in all_signals:
        if hasattr(sig, 'timestamp'):
            ts = sig.timestamp
            if isinstance(ts, pd.Timestamp):
                if ts.tzinfo is None:
                    ts = ts.tz_localize('UTC')
                else:
                    ts = ts.tz_convert('UTC')
                sig_dt = ts.to_pydatetime()
            elif isinstance(ts, datetime):
                if ts.tzinfo is None:
                    sig_dt = ts.replace(tzinfo=timezone.utc)
                else:
                    sig_dt = ts
            else:
                continue
                
            age = (current_time - sig_dt).total_seconds()
            if age <= fresh_threshold:
                fresh_signals += 1
                
    return {
        "symbol": symbol,
        "total_signals": total_signals,
        "fresh_signals": fresh_signals,
        "strategies_enabled": strategies_enabled
    }


def print_signal_report(symbol: str, all_signals: List[Any], log_func=None):
    """
    Выводит подробный отчет о сигналах в лог.
    """
    if not log_func:
        log_func = print
        
    if not all_signals:
        log_func(f"[{symbol}] 📡 СИГНАЛЫ: Нет активных сигналов")
        return

    log_func(f"[{symbol}] 📡 ОТЧЕТ О СИГНАЛАХ ({len(all_signals)}):")
    
    # Группируем по стратегиям
    by_strategy = {}
    for sig in all_signals:
        reason = getattr(sig, 'reason', 'unknown')
        # Определяем тип стратегии по префиксу
        strat = "unknown"
        for prefix in ["ml_", "trend_", "range_", "momentum_", "smc_", "ict_", "zscore_", "vbo_", "amt_of_"]:
            if reason.lower().startswith(prefix):
                strat = prefix.rstrip("_")
                break
        
        if strat not in by_strategy:
            by_strategy[strat] = []
        by_strategy[strat].append(sig)
        
    for strat, signals in by_strategy.items():
        log_func(f"  • {strat.upper()}: {len(signals)} сигналов")
        for i, sig in enumerate(signals[:3]):  # Показываем только первые 3
            action = getattr(sig, 'action', 'HOLD')
            if hasattr(action, 'value'): action = action.value
            price = getattr(sig, 'price', 0)
            ts = getattr(sig, 'timestamp', 'N/A')
            log_func(f"    [{i+1}] {action} @ ${price:.2f} ({getattr(sig, 'reason', '')})")