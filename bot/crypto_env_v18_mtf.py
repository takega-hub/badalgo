"""
МУЛЬТИТАЙМФРЕЙМОВАЯ ВЕРСИЯ СРЕДЫ ОБУЧЕНИЯ V18 MTF
Основана на V17_2_Optimized с добавлением анализа на нескольких таймфреймах:
- 15m: основной таймфрейм торговли
- 1h: фильтр среднесрочного тренда
- 4h: определение основного тренда

ОПТИМИЗИРОВАНО на основе анализа V17:
- Ужесточены фильтры по volatility_ratio (влияние на WR: 52.8%)
- Оптимизирован трейлинг-стоп (снижение % SL_TRAILING с 52% до ~35-40%)
- Улучшены фильтры по ATR, Volume, RSI
- Увеличены TP уровни для лучшего RR
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import csv
import os
from typing import Dict, Tuple, Optional, List, Any
from datetime import datetime
from bot.crypto_env_v17_2_optimized import CryptoTradingEnvV17_2_Optimized
from bot.mtf_optimized_params import (
    MTF_MIN_VOLATILITY_RATIO,
    MTF_MAX_VOLATILITY_RATIO,
    MTF_TRAILING_ACTIVATION_ATR,
    MTF_TRAILING_DISTANCE_ATR,
    MTF_MIN_ABSOLUTE_ATR,
    MTF_ATR_PERCENT_MIN,
    MTF_MIN_ABSOLUTE_VOLUME,
    MTF_MIN_VOLUME_SPIKE,
    MTF_MIN_VOLUME_SPIKE_SHORT,
    MTF_TP_LEVELS,
    MTF_LONG_RSI_MIN,
    MTF_LONG_RSI_MAX,
    MTF_SHORT_RSI_MIN,
    MTF_SHORT_RSI_MAX,
    MTF_MIN_ADX,
    MTF_MIN_ADX_SHORT,
)


class CryptoTradingEnvV18_MTF(CryptoTradingEnvV17_2_Optimized):
    """
    МУЛЬТИТАЙМФРЕЙМОВАЯ ВЕРСИЯ V18
    Расширяет V17_2_Optimized с поддержкой анализа на нескольких таймфреймах
    """
    
    def __init__(self, 
                 df_list: List[pd.DataFrame],  # [df_15m, df_1h, df_4h, ...]
                 obs_cols: List[str],
                 initial_balance: float = 10000.0,
                 commission: float = 0.001,
                 slippage: float = 0.0005,
                 log_file: str = "trades_log_v18_mtf.csv",
                 log_open_positions: bool = False,
                 open_log_file: Optional[str] = None,
                 rr_ratio: float = 2.0,
                 atr_multiplier: float = 2.2,
                 render_mode: Optional[str] = None,
                 training_mode: str = "mtf"):
        """
        Инициализация MTF среды
        
        Args:
            df_list: Список датафреймов [15m, 1h, 4h, ...]
                     Каждый датафрейм должен быть синхронизирован по времени
        """
        # КРИТИЧНО: Устанавливаем MTF атрибуты ДО super().__init__()
        # потому что reset() вызывается в родительском __init__ и использует эти атрибуты
        
        # MTF данные (устанавливаем ДО super().__init__)
        self.df_1h = df_list[1] if len(df_list) > 1 else None
        self.df_4h = df_list[2] if len(df_list) > 2 else None
        self.df_1d = df_list[3] if len(df_list) > 3 else None
        
        # MTF параметры фильтров (устанавливаем ДО super().__init__)
        self.mtf_enabled = self.df_1h is not None and self.df_4h is not None
        self.mtf_strict_mode = True  # Строгий режим: все ТФ должны подтверждать вход
        
        # Кэш для MTF индексов (оптимизация производительности)
        self._mtf_index_cache = {}
        
        # КРИТИЧНО: Устанавливаем observation_space ДО super().__init__()
        # чтобы он был правильного размера когда reset() вызывает _get_observation()
        # ВАЖНО: _count_mtf_features() использует self.df_1h и self.df_4h, которые уже установлены выше
        # Подсчитываем MTF признаки: 6 (1h) + 5 (4h) + 3 (trend alignment) + 1 (conflict) + 4 (zones) = 19
        mtf_features_count = 19 if self.mtf_enabled else 19  # Всегда 19, даже если MTF отключен (заполняются нулями)
        base_features_count = len(obs_cols) + 12  # Базовые признаки + состояние позиции
        n_features = base_features_count + mtf_features_count
        
        # Отладочный вывод
        print(f"[MTF_INIT] Размеры: базовые={base_features_count}, MTF={mtf_features_count}, всего={n_features}")
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features,),
            dtype=np.float32
        )
        
        # Инициализируем базовую среду с основным таймфреймом (15m)
        super().__init__(
            df=df_list[0] if len(df_list) > 0 else pd.DataFrame(),
            obs_cols=obs_cols,
            initial_balance=initial_balance,
            commission=commission,
            slippage=slippage,
            log_file=log_file,
            log_open_positions=log_open_positions,
            open_log_file=open_log_file,
            rr_ratio=rr_ratio,
            atr_multiplier=atr_multiplier,
            render_mode=render_mode,
            training_mode=training_mode
        )
        
        # Переопределяем observation_space после super().__init__() (на случай если родитель изменил его)
        # Но мы уже установили его до super().__init__(), так что это просто для уверенности
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features,),
            dtype=np.float32
        )
        
        # Алиас для основного таймфрейма (устанавливаем после super().__init__)
        self.df_15m = self.df  # Основной таймфрейм (алиас для ясности)
        
        # ========================================================================
        # ПРИМЕНЕНИЕ ОПТИМИЗИРОВАННЫХ ПАРАМЕТРОВ (на основе анализа V17)
        # ========================================================================
        
        # ПРИОРИТЕТ 1: КРИТИЧНЫЕ ОПТИМИЗАЦИИ
        # 1. VOLATILITY_RATIO - САМЫЙ ВАЖНЫЙ ПРИЗНАК (корреляция 0.4182, разница WR 52.8%)
        self.min_volatility_ratio = MTF_MIN_VOLATILITY_RATIO  # 0.0025 (было 0.0020)
        self.max_volatility_ratio = MTF_MAX_VOLATILITY_RATIO  # 1.2 (было 1.5)
        
        # 2. ТРЕЙЛИНГ-СТОП - главная проблема (52% закрытий по SL_TRAILING)
        self.trailing_activation_atr = MTF_TRAILING_ACTIVATION_ATR  # 0.40 (было 0.35)
        self.trailing_distance_atr = MTF_TRAILING_DISTANCE_ATR  # 0.50 (было 0.45)
        
        # ПРИОРИТЕТ 2: ВАЖНЫЕ ОПТИМИЗАЦИИ
        # 3. TP УРОВНИ - для лучшего RR (средний RR 1.69 → цель ≥1.8)
        self.tp_levels = MTF_TP_LEVELS  # [2.5, 3.0, 3.8] (было [2.2, 2.8, 3.6])
        
        # 4. RSI ФИЛЬТРЫ - влияет на WR (разница 48.5%)
        self.long_config['min_rsi_norm'] = MTF_LONG_RSI_MIN  # 0.15 (без изменений)
        self.long_config['max_rsi_norm'] = MTF_LONG_RSI_MAX  # 0.55 (было 0.60)
        self.short_config['min_rsi_norm'] = MTF_SHORT_RSI_MIN  # 0.40 (было 0.35)
        self.short_config['max_rsi_norm'] = MTF_SHORT_RSI_MAX  # 0.85 (без изменений)
        
        # 5. ADX ФИЛЬТРЫ - сила тренда (ОБНОВЛЕНО на основе анализа V18)
        self.min_adx = MTF_MIN_ADX  # 30.0 (было 27.0) - УСИЛЕНО для лучших входов
        self.mtf_min_adx_short = MTF_MIN_ADX_SHORT  # 25.0 (было 22.0) - УСИЛЕНО для SHORT
        
        # Сохраняем оптимизированные параметры для использования в фильтрах (ОБНОВЛЕНО)
        self.mtf_min_absolute_atr = MTF_MIN_ABSOLUTE_ATR  # 150.0 (было 120.0) - УСИЛЕНО
        self.mtf_atr_percent_min = MTF_ATR_PERCENT_MIN  # 0.0015 (без изменений)
        self.mtf_min_absolute_volume = MTF_MIN_ABSOLUTE_VOLUME  # 900.0 (без изменений)
        self.mtf_min_volume_spike = MTF_MIN_VOLUME_SPIKE  # 1.8 (было 1.6) - УСИЛЕНО
        self.mtf_min_volume_spike_short = MTF_MIN_VOLUME_SPIKE_SHORT  # 1.3 (без изменений)
        
        print(f"[MTF_OPTIMIZED] Применены оптимизированные параметры (V18 анализ):")
        print(f"  volatility_ratio: {self.min_volatility_ratio} - {self.max_volatility_ratio}")
        print(f"  trailing: активация={self.trailing_activation_atr}, расстояние={self.trailing_distance_atr}")
        print(f"  TP уровни: {self.tp_levels}")
        print(f"  ATR минимум: {self.mtf_min_absolute_atr}, процент: {self.mtf_atr_percent_min}")
        print(f"  Volume минимум: {self.mtf_min_absolute_volume}, всплеск: {self.mtf_min_volume_spike}/{self.mtf_min_volume_spike_short}")
        print(f"  ADX: LONG={self.min_adx}, SHORT={self.mtf_min_adx_short}")
        
        # Проверка синхронизации данных
        if self.df_1h is not None:
            self._validate_mtf_sync()
    
    def _validate_mtf_sync(self):
        """Проверка синхронизации временных меток между таймфреймами"""
        if self.df_1h is None:
            return
        
        # Проверяем наличие колонки timestamp
        if 'timestamp' not in self.df.columns:
            if isinstance(self.df.index, pd.DatetimeIndex):
                self.df['timestamp'] = self.df.index
            else:
                print("⚠️ [MTF] Предупреждение: нет timestamp в основном датафрейме")
                return
        
        if 'timestamp' not in self.df_1h.columns:
            if isinstance(self.df_1h.index, pd.DatetimeIndex):
                self.df_1h['timestamp'] = self.df_1h.index
            else:
                print("⚠️ [MTF] Предупреждение: нет timestamp в 1h датафрейме")
                return
        
        # Проверяем перекрытие временных диапазонов
        # Правильно конвертируем timestamp (может быть в миллисекундах)
        def safe_to_datetime(ts):
            if isinstance(ts, (int, float)) and ts > 1e12:
                return pd.to_datetime(ts, unit='ms')
            elif isinstance(ts, (int, float)):
                return pd.to_datetime(ts, unit='s')
            else:
                return pd.to_datetime(ts)
        
        df_start = safe_to_datetime(self.df['timestamp'].iloc[0])
        df_end = safe_to_datetime(self.df['timestamp'].iloc[-1])
        df_1h_start = safe_to_datetime(self.df_1h['timestamp'].iloc[0])
        df_1h_end = safe_to_datetime(self.df_1h['timestamp'].iloc[-1])
        
        if df_start < df_1h_start or df_end > df_1h_end:
            print(f"⚠️ [MTF] Предупреждение: временные диапазоны не полностью перекрываются")
            print(f"   15m: {df_start} - {df_end}")
            print(f"   1h:  {df_1h_start} - {df_1h_end}")
    
    def _find_nearest_idx(self, df_target: pd.DataFrame, timestamp: pd.Timestamp) -> int:
        """
        Находит ближайший индекс в целевом датафрейме для заданного timestamp
        
        Args:
            df_target: Целевой датафрейм (1h, 4h и т.д.)
            timestamp: Временная метка из основного датафрейма (15m)
        
        Returns:
            Индекс в целевом датафрейме
        """
        if df_target is None or len(df_target) == 0:
            return 0
        
        # Проверяем кэш
        cache_key = (id(df_target), timestamp)
        if cache_key in self._mtf_index_cache:
            return self._mtf_index_cache[cache_key]
        
        # Определяем колонку с временем
        if 'timestamp' in df_target.columns:
            time_col = 'timestamp'
        elif isinstance(df_target.index, pd.DatetimeIndex):
            time_col = None  # Используем индекс
        else:
            return 0
        
        # Преобразуем timestamp в нужный формат
        def safe_to_datetime(ts):
            if isinstance(ts, pd.Timestamp):
                return ts
            elif isinstance(ts, str):
                return pd.to_datetime(ts)
            elif isinstance(ts, (int, float)) and ts > 1e12:
                return pd.to_datetime(ts, unit='ms')
            elif isinstance(ts, (int, float)):
                return pd.to_datetime(ts, unit='s')
            else:
                return pd.to_datetime(ts)
        
        timestamp = safe_to_datetime(timestamp)
        
        # Ищем ближайший индекс
        if time_col:
            # Правильно конвертируем timestamp колонку
            if df_target[time_col].dtype in ['int64', 'float64', 'int32', 'float32']:
                first_val = df_target[time_col].iloc[0] if len(df_target) > 0 else 0
                if first_val > 1e12:  # Миллисекунды
                    time_series = pd.to_datetime(df_target[time_col], unit='ms')
                else:  # Секунды
                    time_series = pd.to_datetime(df_target[time_col], unit='s')
            else:
                time_series = pd.to_datetime(df_target[time_col])
        else:
            time_series = df_target.index
        
        # Находим ближайший индекс (не больше текущего времени)
        mask = time_series <= timestamp
        if mask.any():
            nearest_idx = mask.idxmax() if isinstance(mask, pd.Series) else np.argmax(mask.values)
            if isinstance(nearest_idx, (int, np.integer)):
                result = int(nearest_idx)
            else:
                # Если индекс не числовой, находим позицию
                result = df_target.index.get_loc(nearest_idx) if hasattr(df_target.index, 'get_loc') else 0
        else:
            result = 0
        
        # Кэшируем результат (ограничиваем размер кэша)
        if len(self._mtf_index_cache) < 1000:
            self._mtf_index_cache[cache_key] = result
        
        return result
    
    def _count_mtf_features(self) -> int:
        """
        Подсчитывает количество MTF признаков
        
        ВАЖНО: Всегда возвращает 19 признаков, даже если MTF отключен.
        Это гарантирует постоянный размер observation_space.
        """
        # Всегда возвращаем фиксированное количество признаков:
        # 6 (1h) + 5 (4h) + 3 (trend alignment) + 1 (conflict) + 4 (zones) = 19
        return 19
    
    def _get_mtf_features(self, current_idx: int) -> np.ndarray:
        """
        Извлекает MTF признаки для текущего шага
        
        Returns:
            Массив MTF признаков
        """
        mtf_features = []
        
        if not self.mtf_enabled:
            return np.zeros(self._count_mtf_features(), dtype=np.float32)
        
        try:
            # Получаем текущее время из основного датафрейма
            # Правильно конвертируем timestamp (может быть в миллисекундах)
            def safe_to_datetime(ts):
                if isinstance(ts, pd.Timestamp):
                    return ts
                elif isinstance(ts, (int, float)) and ts > 1e12:
                    return pd.to_datetime(ts, unit='ms')
                elif isinstance(ts, (int, float)):
                    return pd.to_datetime(ts, unit='s')
                else:
                    return pd.to_datetime(ts)
            
            if 'timestamp' in self.df.columns:
                ts_val = self.df.iloc[current_idx]['timestamp']
                current_time = safe_to_datetime(ts_val)
            elif isinstance(self.df.index, pd.DatetimeIndex):
                current_time = self.df.index[current_idx]
            else:
                return np.zeros(self._count_mtf_features(), dtype=np.float32)
            
            # 1. Признаки с 1h таймфрейма
            if self.df_1h is not None:
                idx_1h = self._find_nearest_idx(self.df_1h, current_time)
                if 0 <= idx_1h < len(self.df_1h):
                    row_1h = self.df_1h.iloc[idx_1h]
                    
                    # Нормализуем признаки
                    adx_1h = float(row_1h.get('adx', 25)) / 100.0
                    plus_di_1h = float(row_1h.get('plus_di', 25)) / 100.0
                    minus_di_1h = float(row_1h.get('minus_di', 25)) / 100.0
                    rsi_1h = float(row_1h.get('rsi', 50)) / 100.0
                    close_1h = float(row_1h.get('close', 0))
                    atr_1h = float(row_1h.get('atr', 0))
                    volume_1h = float(row_1h.get('volume', 0))
                    
                    atr_pct_1h = (atr_1h / close_1h) if close_1h > 0 else 0.0
                    volume_1h_norm = volume_1h / 1000000.0  # Нормализация объема
                    
                    mtf_features.extend([
                        adx_1h,
                        plus_di_1h,
                        minus_di_1h,
                        rsi_1h,
                        atr_pct_1h,
                        volume_1h_norm
                    ])
                else:
                    mtf_features.extend([0.0] * 6)
            
            # 2. Признаки с 4h таймфрейма
            if self.df_4h is not None:
                idx_4h = self._find_nearest_idx(self.df_4h, current_time)
                if 0 <= idx_4h < len(self.df_4h):
                    row_4h = self.df_4h.iloc[idx_4h]
                    # Получаем row_1h если доступен
                    if self.df_1h is not None:
                        idx_1h = self._find_nearest_idx(self.df_1h, current_time)
                        row_1h = self.df_1h.iloc[idx_1h] if 0 <= idx_1h < len(self.df_1h) else None
                    else:
                        row_1h = None
                    
                    adx_4h = float(row_4h.get('adx', 25)) / 100.0
                    plus_di_4h = float(row_4h.get('plus_di', 25)) / 100.0
                    minus_di_4h = float(row_4h.get('minus_di', 25)) / 100.0
                    rsi_4h = float(row_4h.get('rsi', 50)) / 100.0
                    close_4h = float(row_4h.get('close', 0))
                    close_1h_val = float(row_1h.get('close', close_4h)) if row_1h is not None else close_4h
                    
                    price_ratio_4h_1h = (close_4h / close_1h_val) if close_1h_val > 0 else 1.0
                    
                    mtf_features.extend([
                        adx_4h,
                        plus_di_4h,
                        minus_di_4h,
                        rsi_4h,
                        price_ratio_4h_1h
                    ])
                else:
                    mtf_features.extend([0.0] * 5)
            
            # 3. Мультитаймфреймовые соотношения
            trend_alignment = self._calculate_trend_alignment(current_time)
            mtf_features.extend(trend_alignment)
            
            # 4. Конфликтные индикаторы
            conflict_score = self._calculate_conflict_score(current_time)
            mtf_features.append(conflict_score)
            
            # 5. Зональный анализ
            zone_analysis = self._analyze_zones(current_time)
            mtf_features.extend(zone_analysis)
            
        except Exception as e:
            print(f"⚠️ [MTF] Ошибка извлечения MTF признаков: {e}")
            # Заполняем нулями при ошибке
            mtf_features = [0.0] * self._count_mtf_features()
        
        return np.array(mtf_features, dtype=np.float32)
    
    def _calculate_trend_alignment(self, timestamp: pd.Timestamp) -> List[float]:
        """
        Вычисляет выравнивание трендов между таймфреймами
        
        Returns:
            [alignment_15m_1h, alignment_1h_4h, overall_alignment]
        """
        if not self.mtf_enabled:
            return [0.0, 0.0, 0.0]
        
        try:
            idx_1h = self._find_nearest_idx(self.df_1h, timestamp)
            idx_4h = self._find_nearest_idx(self.df_4h, timestamp) if self.df_4h is not None else None
            
            # Тренд на 15m (из текущего шага)
            current_idx = self.current_step
            if current_idx < len(self.df):
                row_15m = self.df.iloc[current_idx]
                plus_di_15m = float(row_15m.get('plus_di', 25))
                minus_di_15m = float(row_15m.get('minus_di', 25))
                trend_15m = 1.0 if plus_di_15m > minus_di_15m else -1.0
            else:
                trend_15m = 0.0
            
            # Тренд на 1h
            if 0 <= idx_1h < len(self.df_1h):
                row_1h = self.df_1h.iloc[idx_1h]
                plus_di_1h = float(row_1h.get('plus_di', 25))
                minus_di_1h = float(row_1h.get('minus_di', 25))
                trend_1h = 1.0 if plus_di_1h > minus_di_1h else -1.0
            else:
                trend_1h = 0.0
            
            # Тренд на 4h
            if idx_4h is not None and 0 <= idx_4h < len(self.df_4h):
                row_4h = self.df_4h.iloc[idx_4h]
                plus_di_4h = float(row_4h.get('plus_di', 25))
                minus_di_4h = float(row_4h.get('minus_di', 25))
                trend_4h = 1.0 if plus_di_4h > minus_di_4h else -1.0
            else:
                trend_4h = 0.0
            
            # Выравнивание между таймфреймами
            alignment_15m_1h = trend_15m * trend_1h  # 1.0 = совпадение, -1.0 = конфликт
            alignment_1h_4h = trend_1h * trend_4h if trend_4h != 0 else 0.0
            
            # Общее выравнивание (все три ТФ в одном направлении)
            if trend_4h != 0:
                overall_alignment = (trend_15m * trend_1h * trend_4h) / 3.0
            else:
                overall_alignment = alignment_15m_1h / 2.0
            
            return [alignment_15m_1h, alignment_1h_4h, overall_alignment]
            
        except Exception as e:
            return [0.0, 0.0, 0.0]
    
    def _calculate_conflict_score(self, timestamp: pd.Timestamp) -> float:
        """
        Вычисляет оценку конфликта трендов между таймфреймами
        
        Returns:
            Score от -1.0 (сильный конфликт) до 1.0 (полное совпадение)
        """
        trend_alignment = self._calculate_trend_alignment(timestamp)
        overall_alignment = trend_alignment[2]
        
        # Преобразуем в конфликтный score (инвертируем)
        conflict_score = -overall_alignment  # -1.0 = конфликт, 1.0 = совпадение
        
        return conflict_score
    
    def _analyze_zones(self, timestamp: pd.Timestamp) -> List[float]:
        """
        Анализ зон перекупленности/перепроданности на разных ТФ
        
        Returns:
            [zone_15m, zone_1h, zone_4h, zone_consensus]
        """
        if not self.mtf_enabled:
            return [0.0, 0.0, 0.0, 0.0]
        
        try:
            # RSI на 15m
            current_idx = self.current_step
            if current_idx < len(self.df):
                rsi_15m = float(self.df.iloc[current_idx].get('rsi', 50))
                zone_15m = (rsi_15m - 50) / 50.0  # Нормализация: -1.0 (перепроданность) до 1.0 (перекупленность)
            else:
                zone_15m = 0.0
            
            # RSI на 1h
            idx_1h = self._find_nearest_idx(self.df_1h, timestamp)
            if 0 <= idx_1h < len(self.df_1h):
                rsi_1h = float(self.df_1h.iloc[idx_1h].get('rsi', 50))
                zone_1h = (rsi_1h - 50) / 50.0
            else:
                zone_1h = 0.0
            
            # RSI на 4h
            idx_4h = self._find_nearest_idx(self.df_4h, timestamp) if self.df_4h is not None else None
            if idx_4h is not None and 0 <= idx_4h < len(self.df_4h):
                rsi_4h = float(self.df_4h.iloc[idx_4h].get('rsi', 50))
                zone_4h = (rsi_4h - 50) / 50.0
            else:
                zone_4h = 0.0
            
            # Консенсус зон (среднее значение)
            zone_consensus = (zone_15m + zone_1h + zone_4h) / 3.0 if zone_4h != 0 else (zone_15m + zone_1h) / 2.0
            
            return [zone_15m, zone_1h, zone_4h, zone_consensus]
            
        except Exception as e:
            return [0.0, 0.0, 0.0, 0.0]
    
    def _get_observation(self) -> np.ndarray:
        """Получение наблюдения с MTF признаками"""
        # Базовое наблюдение из родительского класса
        base_observation = super()._get_observation()
        
        # Добавляем MTF признаки
        mtf_features = self._get_mtf_features(self.current_step)
        
        # Конкатенация
        observation = np.concatenate([base_observation, mtf_features])
        
        # Проверка размера (только при первом вызове для отладки)
        if not hasattr(self, '_obs_size_checked'):
            expected_size = self.observation_space.shape[0]
            actual_size = len(observation)
            if expected_size != actual_size:
                # Автоматически исправляем observation_space
                self.observation_space = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(actual_size,),
                    dtype=np.float32
                )
                # Выводим информативное сообщение только один раз
                print(f"ℹ️ [MTF] Размер observation_space скорректирован: {expected_size} → {actual_size}")
                print(f"   (Базовое: {len(base_observation)}, MTF: {len(mtf_features)})")
            self._obs_size_checked = True
        
        if np.any(np.isnan(observation)):
            observation = np.nan_to_num(observation, nan=0.0)
        
        return observation
    
    def _check_entry_filters_strict(self, price: float, atr: float, action: int = None) -> bool:
        """
        УЖЕСТОЧЕННЫЕ фильтры для входа с MTF проверками и ОПТИМИЗИРОВАННЫМИ параметрами
        
        Переопределяем метод родителя для:
        1. Использования оптимизированных параметров (ATR, Volume, Volatility)
        2. Добавления MTF фильтров
        """
        if self.current_step >= len(self.df):
            return False
        
        try:
            # 1. ОПТИМИЗИРОВАННЫЙ фильтр по волатильности (ATR) - используем MTF параметры
            atr_percent = atr / price
            if atr_percent < self.mtf_atr_percent_min or atr_percent > 0.04:
                return False
            
            # АБСОЛЮТНЫЙ фильтр по ATR (оптимизировано: 120.0 вместо 85.0)
            if atr < self.mtf_min_absolute_atr:
                return False  # Слишком низкий ATR = плохой Win Rate
            
            # 2. Проверка силы тренда через ADX (используем оптимизированные значения)
            if 'adx' in self.df.columns:
                try:
                    adx_value = float(self.df.loc[self.current_step, 'adx'])
                    min_adx_required = self.min_adx  # Уже оптимизировано (27.0)
                    if action == 2:  # SHORT
                        min_adx_required = self.mtf_min_adx_short  # 22.0 (оптимизировано)
                    
                    if adx_value < min_adx_required:
                        return False
                    
                    # Проверка направления через +DI и -DI
                    if action is not None:
                        if 'plus_di' in self.df.columns and 'minus_di' in self.df.columns:
                            try:
                                plus_di = float(self.df.loc[self.current_step, 'plus_di'])
                                minus_di = float(self.df.loc[self.current_step, 'minus_di'])
                                
                                if action == 1:  # LONG
                                    if plus_di <= minus_di:
                                        return False
                                elif action == 2:  # SHORT
                                    if minus_di <= plus_di * 0.95:
                                        return False
                            except:
                                pass
                except:
                    return False
            else:
                atr_percent = atr / price
                if atr_percent < self.mtf_atr_percent_min:
                    return False
            
            # 3. РАЗДЕЛЬНАЯ ПРОВЕРКА RSI (используем оптимизированные значения из long_config/short_config)
            if 'rsi_norm' in self.df.columns:
                try:
                    rsi_norm = float(self.df.loc[self.current_step, 'rsi_norm'])
                    rsi_norm_abs = abs(rsi_norm)
                    
                    if action == 1:  # LONG
                        config = self.long_config  # Уже оптимизировано (max_rsi_norm = 0.55)
                        if rsi_norm_abs < config['min_rsi_norm'] or rsi_norm_abs > config['max_rsi_norm']:
                            return False
                    elif action == 2:  # SHORT
                        config = self.short_config  # Уже оптимизировано (min_rsi_norm = 0.40)
                        if rsi_norm_abs < config['min_rsi_norm'] or rsi_norm_abs > config['max_rsi_norm']:
                            return False
                    else:
                        if rsi_norm_abs < 0.15 or rsi_norm_abs > 0.85:
                            return False
                except:
                    pass
            
            # 4. ОПТИМИЗИРОВАННАЯ проверка объема (используем MTF параметры)
            if 'volume' in self.df.columns:
                try:
                    current_volume = float(self.df.loc[self.current_step, 'volume'])
                    
                    # АБСОЛЮТНЫЙ фильтр (оптимизировано: 900.0 вместо 800.0)
                    if current_volume < self.mtf_min_absolute_volume:
                        return False
                    
                    # ОТНОСИТЕЛЬНЫЙ фильтр (всплеск)
                    if self.current_step >= 20:
                        avg_volume = float(self.df.loc[self.current_step-20:self.current_step, 'volume'].mean())
                        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                        min_spike = self.mtf_min_volume_spike  # 1.6 (оптимизировано)
                        if action == 2:  # SHORT
                            min_spike = self.mtf_min_volume_spike_short  # 1.3 (оптимизировано)
                        if volume_ratio < min_spike:
                            return False
                except:
                    return False
            
            # 5. Проверка движения цены от экстремума
            if self.current_step >= 10:
                try:
                    current_price = float(self.df.loc[self.current_step, 'close'])
                    recent_high = float(self.df.loc[self.current_step-10:self.current_step, 'high'].max())
                    recent_low = float(self.df.loc[self.current_step-10:self.current_step, 'low'].min())
                    
                    if action == 1:  # LONG
                        distance_from_low = ((current_price - recent_low) / recent_low) * 100
                        if distance_from_low < self.min_price_distance_pct:
                            return False
                    elif action == 2:  # SHORT
                        distance_from_high = ((recent_high - current_price) / recent_high) * 100
                        if distance_from_high < self.min_price_distance_pct_short:
                            return False
                except:
                    pass
            
            # 6. ОПТИМИЗИРОВАННАЯ проверка volatility_ratio (используем оптимизированные значения)
            if 'volatility_ratio' in self.df.columns:
                try:
                    volatility_ratio = float(self.df.loc[self.current_step, 'volatility_ratio'])
                    if volatility_ratio < self.min_volatility_ratio:  # 0.0025 (оптимизировано)
                        return False
                    if volatility_ratio > self.max_volatility_ratio:  # 1.2 (оптимизировано)
                        return False
                except:
                    return False
            
            # 7. ГАРАНТИЯ MIN RR RATIO 1.5
            sl_distance = max(atr * self.atr_multiplier, price * self.min_sl_percent)
            sl_distance = min(sl_distance, price * self.max_sl_percent)
            
            min_tp_for_rr = sl_distance * self.min_rr_ratio
            
            min_tp_distance = max(
                min_tp_for_rr,
                atr * self.tp_levels[0],  # Используем оптимизированные TP уровни [2.5, 3.0, 3.8]
                price * self.min_tp_percent
            )
            
            actual_rr = min_tp_distance / sl_distance if sl_distance > 0 else 0
            
            if actual_rr < self.min_rr_ratio:
                self.min_rr_violations += 1
                if self.min_rr_violations % 20 == 0:
                    print(f"[FILTER] RR violation {self.min_rr_violations}: {actual_rr:.2f} < {self.min_rr_ratio}")
                return False
            
            # 8. Дополнительная проверка: TP должен быть достижим
            tp_percent_needed = min_tp_distance / price
            max_tp_pct = 0.02
            if action == 2:
                max_tp_pct = 0.03
            if tp_percent_needed > max_tp_pct:
                return False
            
            # Сохраняем RR статистику
            self.rr_stats.append(actual_rr)
            if len(self.rr_stats) > 100:
                self.rr_stats.pop(0)
            
            # 9. MTF фильтры (если включены)
            if self.mtf_enabled:
                if not self._check_mtf_entry_filters(action):
                    return False
            
            return True
            
        except Exception as e:
            print(f"⚠️ Ошибка в фильтрах входа: {e}")
            return False
    
    def _check_mtf_entry_filters(self, action: int) -> bool:
        """
        MTF фильтры входа
        
        Args:
            action: 1 = LONG, 2 = SHORT
        
        Returns:
            True если все MTF фильтры пройдены
        """
        if not self.mtf_enabled:
            return True
        
        try:
            # Получаем текущее время (правильно конвертируем timestamp)
            def safe_to_datetime(ts):
                if isinstance(ts, pd.Timestamp):
                    return ts
                elif isinstance(ts, (int, float)) and ts > 1e12:
                    return pd.to_datetime(ts, unit='ms')
                elif isinstance(ts, (int, float)):
                    return pd.to_datetime(ts, unit='s')
                else:
                    return pd.to_datetime(ts)
            
            if 'timestamp' in self.df.columns:
                ts_val = self.df.iloc[self.current_step]['timestamp']
                current_time = safe_to_datetime(ts_val)
            elif isinstance(self.df.index, pd.DatetimeIndex):
                current_time = self.df.index[self.current_step]
            else:
                return True  # Если нет времени, пропускаем MTF фильтры
            
            action_type = 'LONG' if action == 1 else 'SHORT' if action == 2 else None
            if action_type is None:
                return True
            
            # 1. Фильтр по тренду на 1h
            if not self._check_1h_trend(current_time, action_type):
                return False
            
            # 2. Фильтр по тренду на 4h (менее строгий)
            if not self._check_4h_trend(current_time, action_type):
                return False
            
            # 3. Проверка конфликта трендов (строгий фильтр)
            if self._check_trend_conflict(current_time, action_type):
                return False
            
            # 4. Поддержка/сопротивление на старших ТФ (опционально, можно добавить позже)
            # if not self._check_mtf_support_resistance(current_time, action_type):
            #     return False
            
            return True
            
        except Exception as e:
            print(f"⚠️ [MTF] Ошибка в MTF фильтрах: {e}")
            return True  # При ошибке пропускаем MTF фильтры
    
    def _check_1h_trend(self, timestamp: pd.Timestamp, action_type: str) -> bool:
        """
        Фильтр по тренду на 1h
        
        Args:
            timestamp: Временная метка
            action_type: 'LONG' или 'SHORT'
        
        Returns:
            True если тренд на 1h поддерживает вход
        """
        if self.df_1h is None:
            return True
        
        try:
            idx_1h = self._find_nearest_idx(self.df_1h, timestamp)
            if not (0 <= idx_1h < len(self.df_1h)):
                return True  # Если нет данных, пропускаем
            
            row_1h = self.df_1h.iloc[idx_1h]
            adx_1h = float(row_1h.get('adx', 0))
            plus_di_1h = float(row_1h.get('plus_di', 25))
            minus_di_1h = float(row_1h.get('minus_di', 25))
            
            if action_type == 'LONG':
                # Для LONG: восходящий тренд на 1h (используем оптимизированный min_adx - 5 для 1h)
                # Новое значение: 30.0 - 5 = 25.0 (более строгий фильтр)
                min_adx_1h = max(25.0, self.min_adx - 5.0)  # 30 - 5 = 25 (усилено)
                return (adx_1h >= min_adx_1h and plus_di_1h > minus_di_1h * 1.15)  # Усилено: 1.1 → 1.15
            else:  # SHORT
                # Для SHORT: нисходящий тренд на 1h (используем оптимизированный min_adx_short - 2 для 1h)
                # Новое значение: 25.0 - 2 = 23.0 (более строгий фильтр)
                min_adx_1h_short = max(23.0, self.mtf_min_adx_short - 2.0)  # 25 - 2 = 23 (усилено)
                return (adx_1h >= min_adx_1h_short and minus_di_1h > plus_di_1h * 1.15)  # Усилено: 1.1 → 1.15
                
        except Exception as e:
            return True  # При ошибке пропускаем
    
    def _check_4h_trend(self, timestamp: pd.Timestamp, action_type: str) -> bool:
        """
        Фильтр по тренду на 4h (направление, но не сила)
        
        Args:
            timestamp: Временная метка
            action_type: 'LONG' или 'SHORT'
        
        Returns:
            True если направление тренда на 4h поддерживает вход
        """
        if self.df_4h is None:
            return True
        
        try:
            idx_4h = self._find_nearest_idx(self.df_4h, timestamp)
            if not (0 <= idx_4h < len(self.df_4h)):
                return True  # Если нет данных, пропускаем
            
            row_4h = self.df_4h.iloc[idx_4h]
            plus_di_4h = float(row_4h.get('plus_di', 25))
            minus_di_4h = float(row_4h.get('minus_di', 25))
            
            # На 4h проверяем только направление (менее строго)
            if action_type == 'LONG':
                return plus_di_4h > minus_di_4h
            else:  # SHORT
                return minus_di_4h > plus_di_4h
                
        except Exception as e:
            return True  # При ошибке пропускаем
    
    def _check_trend_conflict(self, timestamp: pd.Timestamp, action_type: str) -> bool:
        """
        УСИЛЕННАЯ проверка конфликта трендов между ТФ
        
        Args:
            timestamp: Временная метка
            action_type: 'LONG' или 'SHORT'
        
        Returns:
            True если есть конфликт трендов (вход запрещен)
        """
        if not self.mtf_enabled:
            return False
        
        try:
            idx_1h = self._find_nearest_idx(self.df_1h, timestamp) if self.df_1h is not None else None
            idx_4h = self._find_nearest_idx(self.df_4h, timestamp) if self.df_4h is not None else None
            
            if idx_1h is None or idx_4h is None:
                return False  # Если нет данных, не считаем конфликтом
            
            if not (0 <= idx_1h < len(self.df_1h) and 0 <= idx_4h < len(self.df_4h)):
                return False
            
            row_1h = self.df_1h.iloc[idx_1h]
            row_4h = self.df_4h.iloc[idx_4h]
            
            plus_di_1h = float(row_1h.get('plus_di', 25))
            minus_di_1h = float(row_1h.get('minus_di', 25))
            plus_di_4h = float(row_4h.get('plus_di', 25))
            minus_di_4h = float(row_4h.get('minus_di', 25))
            adx_1h = float(row_1h.get('adx', 0))
            adx_4h = float(row_4h.get('adx', 0))
            
            # УСИЛЕННАЯ ПРОВЕРКА: конфликт если:
            # 1. Оба ТФ против входа (базовый случай)
            # 2. Один ТФ против входа И другой нейтрален (слабый сигнал)
            # 3. Оба ТФ нейтральны (нет четкого направления)
            
            if action_type == 'LONG':
                # 1h против LONG
                trend_1h_against = minus_di_1h > plus_di_1h * 1.05  # Усилено: требуется явное превосходство
                # 4h против LONG
                trend_4h_against = minus_di_4h > plus_di_4h * 1.05
                # 1h нейтрален (нет четкого направления)
                trend_1h_neutral = abs(plus_di_1h - minus_di_1h) < 2.0 or adx_1h < 20
                # 4h нейтрален
                trend_4h_neutral = abs(plus_di_4h - minus_di_4h) < 2.0 or adx_4h < 20
                
                # Конфликт если:
                # - Оба против
                # - Один против И другой нейтрален
                # - Оба нейтральны (нет четкого направления для LONG)
                return (trend_1h_against and trend_4h_against) or \
                       (trend_1h_against and trend_4h_neutral) or \
                       (trend_4h_against and trend_1h_neutral) or \
                       (trend_1h_neutral and trend_4h_neutral)
            else:  # SHORT
                # 1h против SHORT
                trend_1h_against = plus_di_1h > minus_di_1h * 1.05
                # 4h против SHORT
                trend_4h_against = plus_di_4h > minus_di_4h * 1.05
                # 1h нейтрален
                trend_1h_neutral = abs(plus_di_1h - minus_di_1h) < 2.0 or adx_1h < 20
                # 4h нейтрален
                trend_4h_neutral = abs(plus_di_4h - minus_di_4h) < 2.0 or adx_4h < 20
                
                # Конфликт если:
                # - Оба против
                # - Один против И другой нейтрален
                # - Оба нейтральны (нет четкого направления для SHORT)
                return (trend_1h_against and trend_4h_against) or \
                       (trend_1h_against and trend_4h_neutral) or \
                       (trend_4h_against and trend_1h_neutral) or \
                       (trend_1h_neutral and trend_4h_neutral)
                
        except Exception as e:
            return False  # При ошибке не считаем конфликтом
    
    def calculate_mtf_tp_sl(self, entry_price: float, action_type: str, current_time: pd.Timestamp) -> Dict[str, float]:
        """
        Расчет TP/SL с учетом мультитаймфреймового контекста
        
        Args:
            entry_price: Цена входа
            action_type: 'LONG' или 'SHORT'
            current_time: Временная метка входа
        
        Returns:
            Словарь с TP, SL и метаданными
        """
        # Базовые TP/SL из родительского класса
        current_atr = float(self.df.iloc[self.current_step].get('atr', entry_price * 0.01))
        
        # Рассчитываем базовые уровни
        sl_distance = max(current_atr * self.atr_multiplier, entry_price * self.min_sl_percent)
        sl_distance = min(sl_distance, entry_price * self.max_sl_percent)
        
        base_tp_distance = max(
            sl_distance * self.min_rr_ratio,
            current_atr * self.tp_levels[0],
            entry_price * self.min_tp_percent
        )
        
        if action_type == 'LONG':
            base_sl = entry_price - sl_distance
            base_tp = entry_price + base_tp_distance
        else:  # SHORT
            base_sl = entry_price + sl_distance
            base_tp = entry_price - base_tp_distance
        
        # MTF анализ для корректировки
        if self.mtf_enabled:
            try:
                # Анализ волатильности на разных ТФ
                idx_1h = self._find_nearest_idx(self.df_1h, current_time) if self.df_1h is not None else None
                idx_4h = self._find_nearest_idx(self.df_4h, current_time) if self.df_4h is not None else None
                
                atr_1h = 0.0
                atr_4h = 0.0
                
                if idx_1h is not None and 0 <= idx_1h < len(self.df_1h):
                    atr_1h = float(self.df_1h.iloc[idx_1h].get('atr', current_atr))
                
                if idx_4h is not None and 0 <= idx_4h < len(self.df_4h):
                    atr_4h = float(self.df_4h.iloc[idx_4h].get('atr', current_atr))
                
                # Средневзвешенный ATR (70% 15m, 20% 1h, 10% 4h)
                weighted_atr = (current_atr * 0.7 + atr_1h * 0.2 + atr_4h * 0.1) if (atr_1h > 0 or atr_4h > 0) else current_atr
                
                # Корректировка TP/SL на основе консенсуса трендов
                trend_strength = self._get_trend_strength_consensus(current_time, action_type)
                
                if trend_strength > 0.8:  # Сильный консенсус
                    tp_multiplier = 1.2
                    sl_multiplier = 0.9
                elif trend_strength > 0.6:  # Средний консенсус
                    tp_multiplier = 1.1
                    sl_multiplier = 0.95
                else:  # Слабый/конфликтный
                    tp_multiplier = 1.0
                    sl_multiplier = 1.1
                
                # Применяем корректировки
                if action_type == 'LONG':
                    new_tp = entry_price + (base_tp - entry_price) * tp_multiplier
                    new_sl = entry_price - (entry_price - base_sl) * sl_multiplier
                else:  # SHORT
                    new_tp = entry_price - (entry_price - base_tp) * tp_multiplier
                    new_sl = entry_price + (base_sl - entry_price) * sl_multiplier
                
                # Гарантируем минимальный RR 1.5
                if action_type == 'LONG':
                    risk = entry_price - new_sl
                    reward = new_tp - entry_price
                else:
                    risk = new_sl - entry_price
                    reward = entry_price - new_tp
                
                rr_ratio = reward / risk if risk > 0 else 0
                
                if rr_ratio < self.min_rr_ratio:
                    # Корректируем TP для достижения RR 1.5
                    required_tp_distance = risk * self.min_rr_ratio
                    if action_type == 'LONG':
                        new_tp = entry_price + required_tp_distance
                    else:
                        new_tp = entry_price - required_tp_distance
                
                return {
                    'tp': new_tp,
                    'sl': new_sl,
                    'weighted_atr': weighted_atr,
                    'trend_strength': trend_strength,
                    'tp_multiplier': tp_multiplier,
                    'sl_multiplier': sl_multiplier
                }
                
            except Exception as e:
                print(f"⚠️ [MTF] Ошибка расчета MTF TP/SL: {e}")
        
        # Возвращаем базовые значения
        return {
            'tp': base_tp,
            'sl': base_sl,
            'weighted_atr': current_atr,
            'trend_strength': 0.5,
            'tp_multiplier': 1.0,
            'sl_multiplier': 1.0
        }
    
    def _get_trend_strength_consensus(self, timestamp: pd.Timestamp, action_type: str) -> float:
        """
        Вычисляет силу консенсуса трендов между таймфреймами
        
        Returns:
            Score от 0.0 (конфликт) до 1.0 (полный консенсус)
        """
        if not self.mtf_enabled:
            return 0.5
        
        try:
            trend_alignment = self._calculate_trend_alignment(timestamp)
            overall_alignment = trend_alignment[2]
            
            # Преобразуем в score консенсуса (0.0 - 1.0)
            consensus_score = (overall_alignment + 1.0) / 2.0  # Нормализация от [-1, 1] к [0, 1]
            
            return consensus_score
            
        except Exception as e:
            return 0.5
    
    def _calculate_reward_profit_focused(self, prev_net_worth: float, 
                                      trade_opened: bool, 
                                      trade_closed: bool,
                                      partial_close: bool,
                                      current_price: float,
                                      action: int) -> float:
        """
        Улучшенная reward функция с MTF бонусами
        
        Переопределяем для добавления MTF наград/штрафов
        """
        # Базовая reward из родительского класса
        base_reward = super()._calculate_reward_profit_focused(
            prev_net_worth, trade_opened, trade_closed, partial_close, current_price, action
        )
        
        # MTF бонусы/штрафы
        mtf_bonus = 0.0
        
        if trade_opened and self.mtf_enabled:
            try:
                # Получаем время открытия
                if 'timestamp' in self.df.columns:
                    entry_time = pd.to_datetime(self.df.iloc[self.current_step]['timestamp'])
                elif isinstance(self.df.index, pd.DatetimeIndex):
                    entry_time = self.df.index[self.current_step]
                else:
                    return base_reward
                
                # 1. Бонус за вход в начале тренда на 4h
                if self._entered_at_trend_start(entry_time):
                    mtf_bonus += 3.0
                
                # 2. Бонус за отсутствие конфликта трендов
                action_type = 'LONG' if action == 1 else 'SHORT' if action == 2 else None
                if action_type and not self._check_trend_conflict(entry_time, action_type):
                    mtf_bonus += 2.0
                
                # 3. Бонус за совпадение зон перекупленности/перепроданности
                zone_alignment = self._get_zone_alignment(entry_time, action_type)
                mtf_bonus += zone_alignment * 1.5
                
                # 4. Штраф за игнорирование сильного тренда на старших ТФ
                if self._ignored_strong_trend(entry_time, action_type):
                    mtf_bonus -= 4.0
                    
            except Exception as e:
                pass  # При ошибке просто возвращаем базовую reward
        
        return base_reward + mtf_bonus
    
    def _entered_at_trend_start(self, timestamp: pd.Timestamp) -> bool:
        """Проверка, был ли вход в начале тренда на 4h"""
        if self.df_4h is None:
            return False
        
        try:
            idx_4h = self._find_nearest_idx(self.df_4h, timestamp)
            if not (0 <= idx_4h < len(self.df_4h)):
                return False
            
            # Проверяем, был ли недавний разворот тренда на 4h
            # (упрощенная проверка: ADX растет, что указывает на начало нового тренда)
            if idx_4h >= 5:
                row_current = self.df_4h.iloc[idx_4h]
                row_prev = self.df_4h.iloc[idx_4h - 5]
                
                adx_current = float(row_current.get('adx', 0))
                adx_prev = float(row_prev.get('adx', 0))
                
                # ADX растет = начало нового тренда
                return adx_current > adx_prev * 1.1
                
        except Exception as e:
            return False
    
    def _get_zone_alignment(self, timestamp: pd.Timestamp, action_type: Optional[str]) -> float:
        """
        Вычисляет выравнивание зон перекупленности/перепроданности
        
        Returns:
            Score от 0.0 (конфликт) до 1.0 (полное совпадение)
        """
        if not self.mtf_enabled or action_type is None:
            return 0.5
        
        try:
            zones = self._analyze_zones(timestamp)
            zone_consensus = zones[3]  # Общий консенсус
            
            # Для LONG: нужна перепроданность (отрицательные значения)
            # Для SHORT: нужна перекупленность (положительные значения)
            if action_type == 'LONG':
                alignment = max(0.0, -zone_consensus)  # Перепроданность = положительный score
            else:  # SHORT
                alignment = max(0.0, zone_consensus)  # Перекупленность = положительный score
            
            return alignment
            
        except Exception as e:
            return 0.5
    
    def _ignored_strong_trend(self, timestamp: pd.Timestamp, action_type: Optional[str]) -> bool:
        """
        Проверка, был ли проигнорирован сильный тренд на старших ТФ
        
        Returns:
            True если был сильный тренд против направления входа
        """
        if not self.mtf_enabled or action_type is None:
            return False
        
        try:
            idx_4h = self._find_nearest_idx(self.df_4h, timestamp) if self.df_4h is not None else None
            if idx_4h is None or not (0 <= idx_4h < len(self.df_4h)):
                return False
            
            row_4h = self.df_4h.iloc[idx_4h]
            adx_4h = float(row_4h.get('adx', 0))
            plus_di_4h = float(row_4h.get('plus_di', 25))
            minus_di_4h = float(row_4h.get('minus_di', 25))
            
            # Сильный тренд на 4h (ADX > 30)
            if adx_4h < 30:
                return False
            
            # Проверяем направление
            if action_type == 'LONG':
                # Игнорировали сильный восходящий тренд = плохо
                return plus_di_4h > minus_di_4h * 1.2
            else:  # SHORT
                # Игнорировали сильный нисходящий тренд = плохо
                return minus_di_4h > plus_di_4h * 1.2
                
        except Exception as e:
            return False
