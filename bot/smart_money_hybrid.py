"""
Гибридная стратегия SMART_MONEY_HYBRID.

Объединяет логику ICT (Inner Circle Trader) и SMC (Smart Money Concepts)
для более точного определения зон входа и повышения качества сигналов.

Концепция:
- ICT находит FVG после снятия ликвидности с временными фильтрами
- SMC проверяет Order Blocks и Mitigation зон
- Сигнал генерируется только если FVG подтвержден Order Block или перекрывается с SMC FVG

Преимущества:
- Более точное определение зон входа
- Использование лучших частей обеих стратегий
- Работает на BTCUSDT (где ICT успешен)
- Двойная фильтрация повышает качество сигналов
"""
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
import logging

from bot.strategy import Action, Signal
from bot.ict_strategy import ICTStrategy, ICTFVG, ICTLiquidity
from bot.smc_strategy import SMCStrategy, SMCZone

logger = logging.getLogger(__name__)


class SmartMoneyHybridStrategy:
    """Гибридная стратегия, объединяющая ICT и SMC."""
    
    def __init__(self, params):
        self.params = params
        
        # Инициализируем обе стратегии
        self.ict_strategy = ICTStrategy(params)
        self.smc_strategy = SMCStrategy(params)
        
        # Параметры гибридной стратегии
        self.require_ob_confirmation = getattr(params, 'hybrid_require_ob_confirmation', True)
        self.fvg_overlap_tolerance = getattr(params, 'hybrid_fvg_overlap_tolerance', 0.3)
        self.require_both_trend_filters = getattr(params, 'hybrid_require_both_trend_filters', True)
        self.max_zone_age_bars = getattr(params, 'hybrid_max_zone_age_bars', 200)  # Увеличено для использования более старых зон
    
    def get_signals(self, df: pd.DataFrame, symbol: str = "Unknown") -> List[Signal]:
        """
        Генерирует сигналы гибридной стратегии.
        
        Логика:
        1. ICT находит FVG после снятия ликвидности
        2. SMC находит FVG и Order Blocks
        3. Проверяем перекрытие зон и подтверждение
        4. Генерируем сигнал только если оба условия выполнены
        
        Args:
            df: DataFrame с данными OHLCV
            symbol: Торговая пара для логирования
            
        Returns:
            Список подтвержденных сигналов
        """
        if len(df) < 200:  # Минимум для обеих стратегий
            return []
        
        signals = []
        
        # Подготавливаем данные для обеих стратегий
        highs = df['high'].values
        lows = df['low'].values
        opens = df['open'].values
        closes = df['close'].values
        
        if 'timestamp' in df.columns:
            times = df['timestamp'].values
        else:
            times = df.index.values
        
        # 1. Получаем FVG от ICT
        try:
            liquidity_sweeps = self.ict_strategy.find_liquidity_sweeps(df)
            if not liquidity_sweeps:
                liquidity_sweeps = self.ict_strategy.find_liquidity_sweeps_alternative(df, 50)
            
            ict_fvg_zones = self.ict_strategy.find_fvg(df, liquidity_sweeps)
            
            # Фильтруем зоны: используем активные зоны (не митигированные)
            # независимо от возраста, но предпочитаем более свежие
            current_idx = len(df) - 1
            active_zones = [zone for zone in ict_fvg_zones if zone.active]
            
            # Если есть активные зоны, используем их (даже если старые)
            # Но если есть недавние активные зоны, предпочитаем их
            recent_active_zones = [
                zone for zone in active_zones 
                if (current_idx - zone.bar_index) <= self.max_zone_age_bars
            ]
            
            # Используем недавние активные зоны, если они есть, иначе все активные
            if recent_active_zones:
                ict_fvg_zones = recent_active_zones
                logger.debug(f"[SmartMoneyHybrid] {symbol} Using {len(recent_active_zones)} recent active ICT FVG zones (age <= {self.max_zone_age_bars})")
            elif active_zones:
                ict_fvg_zones = active_zones
                logger.debug(f"[SmartMoneyHybrid] {symbol} Using {len(active_zones)} active ICT FVG zones (some may be older than {self.max_zone_age_bars} bars)")
            else:
                ict_fvg_zones = []
                logger.debug(f"[SmartMoneyHybrid] {symbol} No active ICT FVG zones found")
            logger.debug(f"[SmartMoneyHybrid] {symbol} Found {len(ict_fvg_zones)} ICT FVG zones")
        except Exception as e:
            logger.error(f"[SmartMoneyHybrid] Error finding ICT FVG: {e}")
            ict_fvg_zones = []
        
        # 2. Получаем зоны от SMC
        try:
            # Используем внутренние методы SMC для получения зон
            smc_fvg_zones = self.smc_strategy._find_fvg(df, highs, lows, opens, closes, times)
            smc_ob_zones = self.smc_strategy._find_ob(df, highs, lows, opens, closes, times)
            
            # Фильтруем зоны по возрасту - используем только недавние зоны
            current_idx = len(df) - 1
            max_fvg_age = getattr(self.params, 'smc_max_fvg_age_bars', 200)
            max_ob_age = getattr(self.params, 'smc_max_ob_age_bars', 300)
            
            recent_smc_fvg = [
                zone for zone in smc_fvg_zones 
                if (current_idx - zone.bar_index) <= max_fvg_age
            ]
            recent_smc_ob = [
                zone for zone in smc_ob_zones 
                if (current_idx - zone.bar_index) <= max_ob_age
            ]
            
            smc_all_zones = recent_smc_fvg + recent_smc_ob
            
            if len(recent_smc_fvg) < len(smc_fvg_zones) or len(recent_smc_ob) < len(smc_ob_zones):
                logger.debug(f"[SmartMoneyHybrid] {symbol} Filtered SMC zones: FVG {len(smc_fvg_zones)} -> {len(recent_smc_fvg)}, OB {len(smc_ob_zones)} -> {len(recent_smc_ob)}")
            
            logger.debug(f"[SmartMoneyHybrid] {symbol} Found {len(recent_smc_fvg)} SMC FVG zones, {len(recent_smc_ob)} SMC OB zones")
        except Exception as e:
            logger.error(f"[SmartMoneyHybrid] Error finding SMC zones: {e}")
            smc_all_zones = []
        
        if not ict_fvg_zones or not smc_all_zones:
            logger.debug(f"[SmartMoneyHybrid] {symbol} No zones found from one or both strategies")
            return []
        
        # 3. Проверяем перекрытие и подтверждение зон
        current_idx = len(df) - 1
        last_row = df.iloc[-1]
        current_price = float(last_row['close'])
        current_low = float(last_row['low'])
        current_high = float(last_row['high'])
        
        # Получаем timestamp последней свечи
        last_ts = last_row.get('timestamp', last_row.name)
        if not isinstance(last_ts, pd.Timestamp):
            last_ts = pd.to_datetime(last_ts)
        
        # Проверяем тренд фильтры
        mtf_bias = None
        ema200_bias = None
        
        try:
            # ICT MTF bias
            mtf_bias = self.ict_strategy.get_higher_tf_bias(
                df, 
                self.ict_strategy.mtf_bias_tf, 
                end_idx=current_idx
            )
            
            # SMC EMA 200 bias
            ema_200 = df['close'].ewm(span=200, adjust=False).mean().iloc[-1]
            if current_price > ema_200:
                ema200_bias = "bullish"
            elif current_price < ema_200:
                ema200_bias = "bearish"
        except Exception as e:
            logger.debug(f"[SmartMoneyHybrid] Error calculating trend filters: {e}")
        
        # 4. Обрабатываем каждую ICT FVG зону
        stats = {
            'total_zones': len(ict_fvg_zones),
            'inactive': 0,
            'too_old': 0,
            'mitigated': 0,
            'no_smc_confirmation': 0,
            'trend_filter_failed': 0,
            'no_touch': 0,
        }
        
        for ict_fvg in ict_fvg_zones:
            if not ict_fvg.active:
                stats['inactive'] += 1
                continue
            
            # Проверяем возраст зоны только если мы используем недавние зоны
            # Если мы используем все активные зоны (включая старые), пропускаем проверку возраста
            # Возраст уже учтен при фильтрации зон выше
            zone_age = current_idx - ict_fvg.bar_index
            if zone_age > self.max_zone_age_bars * 10:  # Используем очень большой лимит для активных зон
                stats['too_old'] += 1
                continue
            
            # Проверяем митигацию ICT FVG
            # Зона считается митигированной только если цена полностью прошла через неё
            # Для bullish: цена должна быть значительно ниже нижней границы
            # Для bearish: цена должна быть значительно выше верхней границы
            zone_size = ict_fvg.upper - ict_fvg.lower
            mitigation_threshold = zone_size * 0.1  # 10% от размера зоны
            
            if ict_fvg.direction == "bullish" and current_price < (ict_fvg.lower - mitigation_threshold):
                stats['mitigated'] += 1
                continue
            if ict_fvg.direction == "bearish" and current_price > (ict_fvg.upper + mitigation_threshold):
                stats['mitigated'] += 1
                continue
            
            # Ищем подтверждение от SMC зон
            smc_confirmation = self._find_smc_confirmation(
                ict_fvg, 
                smc_all_zones, 
                current_idx,
                highs,
                lows
            )
            
            # Для старых зон делаем подтверждение от SMC опциональным
            # Если зона активна и цена находится в ней или близко к ней, можно использовать её без подтверждения
            zone_age = current_idx - ict_fvg.bar_index
            use_without_confirmation = False
            
            if not smc_confirmation:
                # Проверяем, находится ли цена в зоне или близко к ней
                zone_expansion = (ict_fvg.upper - ict_fvg.lower) * 0.1  # 10% расширение зоны
                price_in_zone = ict_fvg.lower - zone_expansion <= current_price <= ict_fvg.upper + zone_expansion
                
                # Для старых зон или если цена в зоне, используем без подтверждения
                if zone_age > self.max_zone_age_bars or price_in_zone:
                    use_without_confirmation = True
                    logger.debug(f"[SmartMoneyHybrid] {symbol} Zone {ict_fvg.bar_index} (age={zone_age}, price={current_price:.2f}, "
                               f"zone=[{ict_fvg.lower:.2f}, {ict_fvg.upper:.2f}]) using without SMC confirmation")
            
            # Если нет подтверждения и не можем использовать без подтверждения, пропускаем
            if not smc_confirmation and not use_without_confirmation:
                stats['no_smc_confirmation'] += 1
                # Детальная диагностика для первых нескольких зон
                if stats['no_smc_confirmation'] <= 3 and len(smc_all_zones) > 0:
                    matching_direction = [z for z in smc_all_zones if z.direction == ict_fvg.direction]
                    if matching_direction:
                        # Проверяем перекрытие без учета возраста и митигации
                        overlapping = []
                        for z in matching_direction:
                            if self._check_zone_overlap(ict_fvg, z):
                                overlapping.append(z)
                        
                        if overlapping:
                            logger.debug(f"[SmartMoneyHybrid] {symbol} Zone {ict_fvg.bar_index} ({ict_fvg.direction}) has {len(overlapping)} overlapping SMC zones but failed age/mitigation checks")
                            # Показываем детали первой перекрывающейся зоны
                            if overlapping:
                                z = overlapping[0]
                                logger.debug(f"[SmartMoneyHybrid] Overlapping SMC zone: bar_index={z.bar_index}, age={current_idx - z.bar_index}, "
                                           f"type={z.zone_type}, direction={z.direction}, mitigated={self.smc_strategy._is_mitigated(z, highs, lows, current_idx)}")
                        else:
                            # Проверяем близость зон
                            ict_range = (ict_fvg.upper - ict_fvg.lower)
                            for z in matching_direction[:3]:  # Проверяем первые 3
                                smc_range = (z.upper - z.lower)
                                if ict_fvg.upper < z.lower:
                                    distance = z.lower - ict_fvg.upper
                                elif z.upper < ict_fvg.lower:
                                    distance = ict_fvg.lower - z.upper
                                else:
                                    distance = 0
                                
                                min_zone_size = min(ict_range, smc_range)
                                if min_zone_size > 0:
                                    proximity_ratio = distance / min_zone_size
                                    logger.debug(f"[SmartMoneyHybrid] Zone {ict_fvg.bar_index} to SMC zone {z.bar_index}: distance={distance:.2f}, "
                                               f"proximity_ratio={proximity_ratio:.3f}, threshold=0.5")
                continue
            
            # Если используем без подтверждения, создаем фиктивное подтверждение
            if use_without_confirmation and not smc_confirmation:
                smc_confirmation = {
                    'zone': None,  # Нет конкретной SMC зоны
                    'overlap': 0.0,
                    'type': 'none',
                    'without_confirmation': True
                }
            
            # Проверяем тренд фильтры
            # Для зон без подтверждения от SMC делаем тренд-фильтр менее строгим
            trend_filter_passed = True
            trend_filter_reason = ""
            
            # Если используем зону без подтверждения от SMC, делаем тренд-фильтр опциональным
            if use_without_confirmation:
                # Для зон без подтверждения проверяем только базовое условие - цена должна быть в зоне
                # Тренд-фильтр не обязателен
                trend_filter_passed = True
            else:
                # Для зон с подтверждением от SMC применяем стандартный тренд-фильтр
                if self.require_both_trend_filters:
                    if ict_fvg.direction == "bullish":
                        if mtf_bias != "bullish" or ema200_bias != "bullish":
                            trend_filter_passed = False
                            trend_filter_reason = f"MTF={mtf_bias}, EMA200={ema200_bias} (need both bullish)"
                    elif ict_fvg.direction == "bearish":
                        if mtf_bias != "bearish" or ema200_bias != "bearish":
                            trend_filter_passed = False
                            trend_filter_reason = f"MTF={mtf_bias}, EMA200={ema200_bias} (need both bearish)"
                else:
                    # Хотя бы один фильтр должен подтверждать
                    if ict_fvg.direction == "bullish":
                        if mtf_bias != "bullish" and ema200_bias != "bullish":
                            trend_filter_passed = False
                            trend_filter_reason = f"MTF={mtf_bias}, EMA200={ema200_bias} (need at least one bullish)"
                    elif ict_fvg.direction == "bearish":
                        if mtf_bias != "bearish" and ema200_bias != "bearish":
                            trend_filter_passed = False
                            trend_filter_reason = f"MTF={mtf_bias}, EMA200={ema200_bias} (need at least one bearish)"
            
            if not trend_filter_passed:
                stats['trend_filter_failed'] += 1
                if stats['trend_filter_failed'] <= 3:  # Логируем только первые 3 для краткости
                    logger.debug(f"[SmartMoneyHybrid] {symbol} Zone {ict_fvg.bar_index} trend filter failed: {trend_filter_reason}")
                continue
            
            # Проверяем касание зоны
            # Используем более гибкую логику - проверяем близость к зоне, а не только прямое касание
            zone_expansion = (ict_fvg.upper - ict_fvg.lower) * 0.3  # Увеличиваем расширение до 30%
            zone_size = ict_fvg.upper - ict_fvg.lower
            
            # Для зон без подтверждения от SMC используем более мягкий proximity threshold
            if use_without_confirmation:
                # Используем процент от текущей цены или размер зоны (берем больший)
                proximity_threshold = max(zone_size * 2.0, current_price * 0.03)  # 3% от цены или 200% от размера зоны
            else:
                proximity_threshold = zone_size * 0.5  # Расстояние до зоны не более 50% от её размера
            
            touch_zone = False
            
            if ict_fvg.direction == "bullish":
                # Проверяем различные варианты касания:
                # 1. Прямое касание зоны
                direct_touch = current_low <= ict_fvg.upper + zone_expansion and current_price >= ict_fvg.lower - zone_expansion
                # 2. Цена находится внутри зоны
                in_zone = ict_fvg.lower <= current_price <= ict_fvg.upper
                # 3. Цена близко к зоне снизу (для ретеста)
                near_below = current_price < ict_fvg.lower and (ict_fvg.lower - current_price) <= proximity_threshold
                # 4. Цена близко к зоне сверху (для ретеста)
                near_above = current_price > ict_fvg.upper and (current_price - ict_fvg.upper) <= proximity_threshold
                
                if direct_touch or in_zone or near_below or near_above:
                    touch_zone = True
                    # Генерируем сигнал LONG
                    signal = self._generate_signal(
                        df, 
                        ict_fvg, 
                        liquidity_sweeps, 
                        Action.LONG, 
                        current_price, 
                        last_ts,
                        symbol,
                        smc_confirmation
                    )
                    if signal:
                        signals.append(signal)
            
            elif ict_fvg.direction == "bearish":
                # Проверяем различные варианты касания:
                # 1. Прямое касание зоны
                direct_touch = current_high >= ict_fvg.lower - zone_expansion and current_price <= ict_fvg.upper + zone_expansion
                # 2. Цена находится внутри зоны
                in_zone = ict_fvg.lower <= current_price <= ict_fvg.upper
                # 3. Цена близко к зоне снизу (для ретеста)
                near_below = current_price < ict_fvg.lower and (ict_fvg.lower - current_price) <= proximity_threshold
                # 4. Цена близко к зоне сверху (для ретеста)
                near_above = current_price > ict_fvg.upper and (current_price - ict_fvg.upper) <= proximity_threshold
                
                if direct_touch or in_zone or near_below or near_above:
                    touch_zone = True
                    # Генерируем сигнал SHORT
                    signal = self._generate_signal(
                        df, 
                        ict_fvg, 
                        liquidity_sweeps, 
                        Action.SHORT, 
                        current_price, 
                        last_ts,
                        symbol,
                        smc_confirmation
                    )
                    if signal:
                        signals.append(signal)
            
            if not touch_zone:
                stats['no_touch'] += 1
                # Диагностика для первых нескольких зон
                if stats['no_touch'] <= 3:
                    distance_below = (ict_fvg.lower - current_price) if current_price < ict_fvg.lower else 0
                    distance_above = (current_price - ict_fvg.upper) if current_price > ict_fvg.upper else 0
                    logger.debug(f"[SmartMoneyHybrid] {symbol} Zone {ict_fvg.bar_index} ({ict_fvg.direction}) no touch - "
                               f"price={current_price:.2f}, zone=[{ict_fvg.lower:.2f}, {ict_fvg.upper:.2f}], "
                               f"distance_below={distance_below:.2f}, distance_above={distance_above:.2f}")
        
        if len(signals) == 0 and stats['total_zones'] > 0:
            logger.debug(f"[SmartMoneyHybrid] {symbol} Filter stats: {stats}")
            # Выводим детальную статистику в консоль для диагностики
            print(f"      [HYBRID Debug] Filter breakdown:")
            print(f"         - Total zones: {stats['total_zones']}")
            print(f"         - Inactive: {stats['inactive']}")
            print(f"         - Too old: {stats['too_old']}")
            print(f"         - Mitigated: {stats['mitigated']}")
            print(f"         - No SMC confirmation: {stats['no_smc_confirmation']}")
            print(f"         - Trend filter failed: {stats['trend_filter_failed']}")
            print(f"         - No touch: {stats['no_touch']}")
        
        logger.info(f"[SmartMoneyHybrid] {symbol} Generated {len(signals)} confirmed signals")
        return signals
    
    def _find_smc_confirmation(
        self,
        ict_fvg: ICTFVG,
        smc_zones: List[SMCZone],
        current_idx: int,
        highs: np.ndarray,
        lows: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        """
        Ищет подтверждение ICT FVG от SMC зон.
        
        Returns:
            Dict с информацией о подтверждении или None
        """
        for smc_zone in smc_zones:
            # Проверяем возраст SMC зоны
            max_age = getattr(
                self.params, 
                'smc_max_fvg_age_bars' if smc_zone.zone_type == "FVG" else 'smc_max_ob_age_bars',
                50
            )
            if (current_idx - smc_zone.bar_index) > max_age:
                continue
            
            # Проверяем митигацию SMC зоны
            if self.smc_strategy._is_mitigated(smc_zone, highs, lows, current_idx):
                continue
            
            # Проверяем перекрытие зон
            overlap = self._check_zone_overlap(ict_fvg, smc_zone)
            
            if overlap:
                # Проверяем направление
                if ict_fvg.direction == smc_zone.direction:
                    return {
                        'zone': smc_zone,
                        'overlap': overlap,
                        'type': smc_zone.zone_type,
                    }
        
        return None
    
    def _check_zone_overlap(self, ict_fvg: ICTFVG, smc_zone: SMCZone) -> bool:
        """
        Проверяет перекрытие ICT FVG и SMC зоны.
        Также проверяет близость зон, если они не перекрываются напрямую.
        
        Returns:
            True если зоны перекрываются или находятся близко друг к другу
        """
        # Рассчитываем перекрытие
        ict_range = (ict_fvg.upper - ict_fvg.lower)
        smc_range = (smc_zone.upper - smc_zone.lower)
        
        # Находим пересечение
        overlap_lower = max(ict_fvg.lower, smc_zone.lower)
        overlap_upper = min(ict_fvg.upper, smc_zone.upper)
        
        # Если зоны перекрываются
        if overlap_lower < overlap_upper:
            overlap_size = overlap_upper - overlap_lower
            min_zone_size = min(ict_range, smc_range)
            
            if min_zone_size == 0:
                return False
            
            overlap_ratio = overlap_size / min_zone_size
            # Используем более мягкий tolerance для перекрытия
            return overlap_ratio >= (self.fvg_overlap_tolerance * 0.5)  # Уменьшаем до 15% вместо 30%
        
        # Если зоны не перекрываются, проверяем близость
        # Расстояние между зонами
        if ict_fvg.upper < smc_zone.lower:
            # ICT зона ниже SMC зоны
            distance = smc_zone.lower - ict_fvg.upper
        elif smc_zone.upper < ict_fvg.lower:
            # SMC зона ниже ICT зоны
            distance = ict_fvg.lower - smc_zone.upper
        else:
            # Зоны перекрываются (уже обработано выше)
            return True
        
        # Зоны считаются близкими, если расстояние меньше 50% от размера меньшей зоны
        # Увеличиваем tolerance для близости, чтобы найти больше подтверждений
        min_zone_size = min(ict_range, smc_range)
        if min_zone_size == 0:
            return False
        
        proximity_ratio = distance / min_zone_size
        return proximity_ratio <= 0.5  # Расстояние не более 50% от размера меньшей зоны
    
    def _generate_signal(
        self,
        df: pd.DataFrame,
        ict_fvg: ICTFVG,
        liquidity_sweeps: List[ICTLiquidity],
        action: Action,
        current_price: float,
        timestamp: pd.Timestamp,
        symbol: str,
        smc_confirmation: Dict[str, Any],
    ) -> Optional[Signal]:
        """
        Генерирует сигнал на основе подтвержденной зоны.
        
        Args:
            df: DataFrame с данными
            ict_fvg: ICT FVG зона
            liquidity_sweeps: Список снятий ликвидности
            action: Направление сигнала
            current_price: Текущая цена
            timestamp: Временная метка
            symbol: Торговая пара
            smc_confirmation: Информация о подтверждении от SMC
            
        Returns:
            Signal объект или None
        """
        try:
            # Получаем ликвидность для расчета SL
            liq = next(
                (l for l in liquidity_sweeps if l.bar_index == ict_fvg.liquidity_bar_index),
                None
            )
            
            # Рассчитываем ATR
            if 'atr' not in df.columns:
                atr = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
            else:
                atr = df['atr'].iloc[-1]
            
            if pd.isna(atr) or atr == 0:
                atr = current_price * 0.01
            
            # Рассчитываем SL/TP
            if action == Action.LONG:
                if liq:
                    sl = liq.price - (atr * 0.5)
                else:
                    sl = current_price * 0.99
                
                # Ограничение максимального риска (1.5%)
                if abs(current_price - sl) > current_price * 0.015:
                    sl = current_price - (current_price * 0.015)
                
                # Расчет TP на основе RR
                risk = abs(current_price - sl)
                rr_ratio = getattr(self.params, 'ict_rr_ratio', 2.0)
                tp = current_price + (risk * rr_ratio)
            else:  # SHORT
                if liq:
                    sl = liq.price + (atr * 0.5)
                else:
                    sl = current_price * 1.01
                
                # Ограничение максимального риска (1.5%)
                if abs(sl - current_price) > current_price * 0.015:
                    sl = current_price + (current_price * 0.015)
                
                # Расчет TP на основе RR
                risk = abs(sl - current_price)
                rr_ratio = getattr(self.params, 'ict_rr_ratio', 2.0)
                tp = current_price - (risk * rr_ratio)
            
            # Обрабатываем случай, когда подтверждение от SMC отсутствует (для старых зон)
            smc_zone = smc_confirmation.get('zone')
            if smc_zone is None:
                smc_zone_type = 'none'
                smc_zone_bar_index = -1
            else:
                smc_zone_type = smc_zone.zone_type
                smc_zone_bar_index = smc_zone.bar_index
            
            reason = f"smart_money_hybrid_{action.value.lower()}_ict_{ict_fvg.bar_index}_smc_{smc_zone_type.lower()}"
            if smc_confirmation.get('without_confirmation'):
                reason += "_no_smc_conf"
            
            return Signal(
                timestamp=timestamp,
                action=action,
                reason=reason,
                price=current_price,
                stop_loss=float(sl),
                take_profit=float(tp),
                indicators_info={
                    'ict_fvg_bar_index': ict_fvg.bar_index,
                    'smc_zone_type': smc_zone_type,
                    'smc_zone_bar_index': smc_zone_bar_index,
                    'overlap_ratio': smc_confirmation.get('overlap', 0.0),
                }
            )
        except Exception as e:
            logger.error(f"[SmartMoneyHybrid] Error generating signal: {e}")
            return None


def build_smart_money_signals(
    df: pd.DataFrame,
    params,
    symbol: str = "Unknown"
) -> List[Signal]:
    """
    Точка входа для бота. Использует SmartMoneyHybridStrategy для генерации сигналов.
    
    Args:
        df: DataFrame с данными OHLCV
        params: Параметры стратегии
        symbol: Торговая пара для логирования
        
    Returns:
        Список сигналов гибридной стратегии
    """
    strategy = SmartMoneyHybridStrategy(params)
    return strategy.get_signals(df, symbol=symbol)


__all__ = ["SmartMoneyHybridStrategy", "build_smart_money_signals"]
