import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import csv
import os
from typing import Dict, Tuple, Optional, List, Any
from datetime import datetime
import matplotlib.pyplot as plt


class CryptoTradingEnvV17_Optimized(gym.Env):
    """
    ОПТИМИЗИРОВАННАЯ ВЕРСИЯ ПОСЛЕ АНАЛИЗА
    Ключевые улучшения:
    1. ГАРАНТИРОВАННЫЙ RR RATIO ≥ 1.5
    2. УЖЕСТОЧЕННЫЕ ФИЛЬТРЫ ВХОДА
    3. УЛУЧШЕННЫЕ TP УРОВНИ
    4. ОПТИМИЗИРОВАННЫЙ ТРЕЙЛИНГ-СТОП
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, 
                 df: pd.DataFrame,
                 obs_cols: List[str],
                 initial_balance: float = 10000.0,
                 commission: float = 0.001,
                 slippage: float = 0.0005,
                 log_file: str = "trades_log_v17_optimized.csv",
                 rr_ratio: float = 2.0,
                 atr_multiplier: float = 2.5,
                 max_daily_trades: int = 5,
                 trade_cooldown_steps: int = 10,
                 render_mode: Optional[str] = None,
                 training_mode: str = "optimized"):
        """
        Инициализация оптимизированной среды
        """
        super(CryptoTradingEnvV17_Optimized, self).__init__()
        
        # Валидация и подготовка данных
        self.df = self._prepare_data_simple(df.copy())
        self.obs_cols = obs_cols
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.log_file = log_file
        
        # ОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ
        self.base_rr_ratio = rr_ratio
        self.atr_multiplier = atr_multiplier
        self.min_rr_ratio = 1.5  # ГАРАНТИРОВАННЫЙ МИНИМУМ RR 1.5:1
        
        # TP уровни: делаем более достижимыми (по итогам анализа)
        self.tp_levels = [1.5, 2.2, 3.0]
        # Больше закрытий на первых уровнях (фиксируем часть прибыли раньше)
        self.tp_close_percentages = [0.30, 0.40, 0.30]
        
        # Трейлинг-стоп: менее агрессивный (по отчёту много SL_TRAILING)
        # Базовая дистанция (дальше динамически расширяем при росте прибыли)
        self.trailing_activation_atr = 0.35   # базовая для LONG (для SHORT задаём отдельно в _update_trailing_stop)
        self.trailing_distance_atr = 0.40
        self.protective_trailing_atr = 0.60
        # Время удержания
        self.max_hold_steps = 60
        self.min_hold_steps = 8
        
        # УЖЕСТОЧЕННЫЕ ФИЛЬТРЫ ВХОДА (УЛУЧШЕНО)
        self.min_sl_percent = 0.003           # Минимальный SL 0.3% (оставляем как есть)
        self.max_sl_percent = 0.007           # УМЕНЬШЕНО с 0.008 до 0.007 - более строгий SL

        self.min_tp_percent = 0.006          # УМЕНЬШЕНО с 0.008 до 0.006 - TP уровни уже снижены до [1.8, 2.5, 3.5]
        
        # Маржинальность: консервативная
        self.base_margin_percent = 0.07
        
        # Лимит сделок/кулдаун: в обучении полезно больше сделок для сигнала reward
        self.max_daily_trades = int(max_daily_trades) if max_daily_trades is not None else 5
        self.trade_cooldown_steps = int(trade_cooldown_steps) if trade_cooldown_steps is not None else 10
        self.trades_today = 0
        self.current_day = 0
        
        # УЖЕСТОЧЕННЫЕ ФИЛЬТРЫ ДЛЯ КАЧЕСТВЕННОГО ВХОДА (ПО РЕКОМЕНДАЦИЯМ АНАЛИЗА)
        # Анализ показал: Win Rate 48.4% (цель ≥50%), LONG WR 34.3%, SHORT WR 27.4%
        # Проблемы: много SL_TRAILING (37.9%), много VERY_BAD сделок (28.2%)
        # Используем ADX (Average Directional Index) - стандартный индикатор силы тренда
        # ADX > 25 = сильный тренд, ADX > 30 = очень сильный тренд
        
        # БАЗОВЫЕ ЗНАЧЕНИЯ ФИЛЬТРОВ (баланс качества и количества сделок)
        self.base_min_adx = 20.0  # Оптимальный баланс: достаточно строгий, но не блокирует все сделки
        self.base_min_trend_strength = 0.55
        
        # АДАПТИВНЫЕ ФИЛЬТРЫ: ослабляются при отсутствии сделок
        self.min_adx = self.base_min_adx
        self.min_trend_strength = self.base_min_trend_strength
        
        # Параметры адаптации фильтров
        self.steps_without_trade = 0
        self.max_steps_without_trade = 50  # УСКОРЕНО: после 50 шагов без сделок ослабляем фильтры
        self.filter_relaxation_rate = 0.95  # Коэффициент ослабления (0.95 = ослабление на 5%)
        self.min_filter_values = {
            'min_adx': 10.0,  # СНИЖЕНО: минимальное значение ADX (было 15.0)
            'min_trend_strength': 0.30  # СНИЖЕНО: минимальное значение trend_strength (было 0.40)
        }
        # volume_ratio УБРАН ИЗ ФИЛЬТРОВ (отрицательная корреляция с PnL: -0.0342)
        # min_volume_ratio оставлен для совместимости, но не используется в фильтрах
        # КРИТИЧНО: volatility_ratio показал разницу Win Rate 15.9%! (Q1: 38.5% vs Q4: 54.4%)
        # Анализ показал: прибыльные сделки имеют volatility_ratio = 0.0043, убыточные = 0.0070
        self.min_volatility_ratio = 0.0030    # ОПТИМИЗИРОВАНО: выше среднего убыточных сделок
        self.max_volatility_ratio = 1.6       # РЕКОМЕНДАЦИЯ: уменьшить с 1.8 до 1.6 (защита от экстремальных значений)
        # ДОБАВЛЯЕМ min_volume для фильтра (анализ показал: volume имеет корреляцию 0.1581 с PnL, разница 31.5%)
        self.min_volume_multiplier = 1.3      # УВЕЛИЧЕНО с 1.2 до 1.3-1.4 (рекомендация анализа)
        # КРИТИЧНО: rsi_norm имеет сильную корреляцию 0.2229 с PnL и влияет на WR (разница 36.8%!)
        # Q1 (низкий rsi_norm): WR 41.7%, Q4 (высокий): WR 71.3%
        # РАЗДЕЛЬНЫЕ RSI ФИЛЬТРЫ ДЛЯ LONG/SHORT (ОПТИМИЗИРОВАНО)
        # Для LONG: перепроданность (низкий RSI) - вход в зоне 0.15-0.60
        # Для SHORT: перекупленность (высокий RSI) - вход в зоне 0.55-0.85
        
        # LONG_CONFIG: Баланс качества и количества сделок
        self.long_config = {
            'min_trend_strength': 0.4,           # умеренный фильтр тренда
            'min_rsi_norm': 0.15,                # расширено: перепроданность (RSI ~15-50)
            'max_rsi_norm': 0.50,                # расширено для большего количества сделок
            'trailing_distance_atr': 0.60,       # согласовано с общим трейлингом (шире)
            'position_size_multiplier': 1.0,     # полный размер позиции (уверенность в LONG)
            'min_volume_ratio': 1.1,             # снижено: требовать нормальный объем
        }
        
        # SHORT_CONFIG: Значительно ослаблено для работы балансировщика
        self.short_config = {
            'enabled': True,
            'min_trend_strength': 0.3,          # Сильно ослаблено: даем шортам больше шансов
            'min_rsi_norm': 0.0,                # УБРАНО: не требуем перекупленность (RSI > 50)
            'max_rsi_norm': 1.0,                # УБРАНО: принимаем любой RSI
            'trailing_distance_atr': 0.60,       # шире, чтобы не выбивало шумом
            'position_size_multiplier': 0.6,    # 60% от объема лонга
            'min_volume_ratio': 0.8,            # Сильно снижено: почти любой объем
            'max_percentage_of_portfolio': 30,  
        }
        
        # Дополнительные фильтры по объёму и цене
        # V2: симметрично для LONG/SHORT, иначе LONG режется сильнее
        self.min_volume_spike_long = 1.0        # 1.0 = не требуем всплеск (только не ниже среднего)
        self.min_volume_spike_short = 1.0
        self.min_price_distance_pct = 1.0        # минимальное движение от экстремума (1%)

        # V2: мягкий DI-фильтр (чтобы LONG не пропадали из-за +DI/-DI)
        self.di_direction_margin = 5.0  # УВЕЛИЧЕНО: допускаем большее преимущество "против" направления
        
        # МАСШТАБИРОВАНИЕ ПРИЗНАКОВ (по анализу важности) - УЛУЧШЕНО
        # RSI имеет самую сильную корреляцию 0.2229 и влияет на WR (разница 36.8%)
        self.obs_scaling = {
            'rsi_norm': 3.0,        # УВЕЛИЧЕНО: самый важный признак (корреляция 0.2229, WR разница 36.8%)
            'volume': 2.0,          # увеличено (корреляция 0.1581, разница 31.5%)
            'atr': 1.5,             # увеличено (корреляция 0.1406)
            'close': 1.0,           # стандартный вес (разница 2.2%)
            'volume_ratio': 0.5,    # уменьшено (плохой предиктор, корреляция -0.0342)
        }

        
        # Параметры для reward (УЛУЧШЕНО - усилены штрафы и награды)
        self.tp_bonus_multiplier = 15.0      # УВЕЛИЧЕНО (было 10.0) - большая награда за TP
        self.tp_full_bonus = 20.0             # УВЕЛИЧЕНО (было 15.0) - награда за полный TP
        self.sl_penalty_multiplier = 8.0     # УВЕЛИЧЕНО (было 5.0) - большой штраф за SL
        self.manual_penalty = 4.0             # УВЕЛИЧЕНО (было 3.0)
        self.time_exit_penalty = 2.0          # УВЕЛИЧЕНО (было 1.5)
        
        # Бонус за качественную сделку (УЛУЧШЕНО)
        self.quality_bonus_threshold = 0.015  # 1.5% прибыли
        self.quality_bonus = 12.0             # УВЕЛИЧЕНО (было 8.0) - больше бонус за качество
        
        # Пространства действий и наблюдений
        self.action_space = spaces.Discrete(3)
        
        # Размер наблюдения
        n_features = len(self.obs_cols) + 12
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features,),
            dtype=np.float32
        )
        
        # Статистика
        self.recent_trades_pnl = []
        self.max_recent_trades = 20
        
        # ГАРАНТИРУЕМ создание папки для логов
        log_dir = os.path.dirname(log_file) if os.path.dirname(log_file) else "."
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = log_file
        
        # Инициализация логирования
        self._init_log_file()
        
        # История для анализа
        self.net_worth_history = []
        self.actions_history = []
        self.rewards_history = []
        self.trade_history = []
        
        # Состояние частичных закрытий
        self.partial_closes = []
        self.tp_closed_levels = [False, False, False]
        
        # Счетчики для reward
        self.consecutive_profitable_trades = 0
        self.consecutive_loss_trades = 0
        self.avg_profit_last_10 = 0
        self.trailing_sl_count = 0  # Счетчик трейлинг-SL закрытий
        self.win_streak = 0  # Текущая серия прибыльных сделок
        self.recent_trailing_sl = []  # История трейлинг-SL закрытий (последние 10)
        
        # Статистика RR
        self.rr_stats = []
        self.min_rr_violations = 0
        
        # Счетчики для разнообразия сделок
        self.long_trades_count = 0
        self.short_trades_count = 0
        
        # АДАПТИВНЫЕ ПАРАМЕТРЫ: для динамической настройки фильтров
        self.steps_without_trade = 0
        self.last_trade_step = 0
        
        # Инициализация адаптивных фильтров (сбрасываем к базовым значениям)
        self.min_adx = self.base_min_adx
        self.min_trend_strength = self.base_min_trend_strength
        self.steps_without_trade = 0
        self.last_trade_step = 0
        
        # Сбрасываем флаг экстренного режима
        if hasattr(self, '_emergency_mode_logged'):
            delattr(self, '_emergency_mode_logged')
        
        # Дополнительные фильтры по объёму и цене
        self.min_volume_spike = 1.5             # минимальный всплеск объёма (1.5x среднего)
        self.min_price_distance_pct = 1.0        # минимальное движение от экстремума (1%)
        
        # МАСШТАБИРОВАНИЕ ПРИЗНАКОВ (по анализу важности) - УЛУЧШЕНО
        # RSI имеет самую сильную корреляцию 0.2229 и влияет на WR (разница 36.8%)
        self.obs_scaling = {
            'rsi_norm': 3.0,        # УВЕЛИЧЕНО: самый важный признак (корреляция 0.2229, WR разница 36.8%)
            'volume': 2.0,          # увеличено (корреляция 0.1581, разница 31.5%)
            'atr': 1.5,             # увеличено (корреляция 0.1406)
            'close': 1.0,           # стандартный вес (разница 2.2%)
            'volume_ratio': 0.5,    # уменьшено (плохой предиктор, корреляция -0.0342)
        }
        
        self.reset()
    
    def _prepare_data_simple(self, df: pd.DataFrame) -> pd.DataFrame:
        """Упрощенная подготовка данных"""
        if len(df) == 0:
            print("⚠️  ВНИМАНИЕ: DataFrame пустой!")
            df = pd.DataFrame({
                'open': [100.0, 101.0, 102.0, 103.0, 104.0],
                'high': [101.0, 102.0, 103.0, 104.0, 105.0],
                'low': [99.0, 100.0, 101.0, 102.0, 103.0],
                'close': [100.5, 101.5, 102.5, 103.5, 104.5],
                'volume': [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
                'atr': [1.0, 1.0, 1.0, 1.0, 1.0]
            })
        
        df = df.reset_index(drop=True).copy()
        
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 100.0
        
        if 'atr' not in df.columns:
            print("⚠️ ATR не найден, создаем")
            high_low = df['high'] - df['low']
            df['atr'] = high_low.rolling(window=14, min_periods=1).mean()
            df['atr'] = df['atr'].fillna(df['atr'].mean() if not df['atr'].isnull().all() else 1.0)
        else:
            df['atr'] = df['atr'].fillna(df['atr'].mean() if not df['atr'].isnull().all() else 1.0)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                mean_val = df[col].mean() if not df[col].isnull().all() else 0
                df[col] = df[col].fillna(mean_val)
        
        print(f"📊 Подготовлено данных: {len(df)} строк")
        
        return df
    
    def _init_log_file(self):
        """Надежная инициализация файла логов"""
        try:
            if os.path.exists(self.log_file):
                try:
                    with open(self.log_file, 'r', encoding='utf-8') as f:
                        first_line = f.readline().strip()
                        if first_line and 'step' in first_line.lower():
                            return
                except:
                    pass
            
            with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'step', 'type', 'entry', 'sl_initial', 'sl_current',
                    'tp_levels', 'exit', 'pnl_percent', 'net_worth',
                    'exit_reason', 'duration', 'trailing', 'tp_closed', 'partial_closes',
                    'trade_quality', 'rr_ratio'  # ДОБАВЛЕН RR RATIO
                ])
            print(f"✅ Файл логов создан: {self.log_file}")
            
        except Exception as e:
            print(f"⚠️ Ошибка создания файла логов: {e}")
            alt_log_file = f"logs/trades_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            os.makedirs(os.path.dirname(alt_log_file), exist_ok=True)
            self.log_file = alt_log_file
            self._init_log_file()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Сброс среды"""
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        
        self.position = 0
        self.entry_price = 0.0
        self.current_sl = 0.0
        self.initial_sl = 0.0
        self.tp_prices = []
        self.actual_exit_price = 0.0
        self.active_margin = 0.0
        self.shares_held = 0.0
        self.shares_remaining = 0.0
        
        self.current_step = 0
        self.steps_since_open = 0
        self.steps_since_last_trade = 0
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.exit_type = None
        self.trailing_active = False
        self.highest_profit_pct = 0.0
        self.lowest_profit_pct = 0.0
        
        self.tp_closed_levels = [False, False, False]
        self.partial_closes = []
        
        self.trades_today = 0
        self.current_day = 0
        
        self.net_worth_history = [self.net_worth]
        self.actions_history = []
        self.rewards_history = []
        self.trade_history = []
        self.recent_trades_pnl = []
        self.num_timesteps = 0  # Счетчик шагов для логирования
        
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.tp_count = 0
        self.sl_count = 0
        self.time_exit_count = 0
        self.manual_count = 0
        
        # Счетчики для разнообразия сделок
        self.long_trades_count = 0
        self.short_trades_count = 0
        
        # Счетчики качества
        self.consecutive_profitable_trades = 0
        self.consecutive_loss_trades = 0
        self.avg_profit_last_10 = 0
        self.trailing_sl_count = 0  # Счетчик трейлинг-SL закрытий
        self.win_streak = 0  # Текущая серия прибыльных сделок
        self.recent_trailing_sl = []  # История трейлинг-SL закрытий (последние 10)
        self.num_timesteps = 0  # Счетчик шагов для логирования
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """Получение наблюдения с индикатором качества"""
        if len(self.df) == 0:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1
        
        # Рыночные данные с масштабированием важных признаков
        try:
            market_data_list = []
            for col in self.obs_cols:
                if col in self.df.columns:
                    try:
                        value = float(self.df.loc[self.current_step, col])
                        if pd.isna(value):
                            value = 0.0
                        # Применяем масштабирование для важных признаков
                        scale = self.obs_scaling.get(col, 1.0)
                        value = value * scale
                        market_data_list.append(value)
                    except:
                        market_data_list.append(0.0)
                else:
                    market_data_list.append(0.0)
            
            market_data = np.array(market_data_list, dtype=np.float32)
        except Exception as e:
            market_data = np.zeros(len(self.obs_cols), dtype=np.float32)
        
        # Расчет индикатора качества сделки
        trade_quality = 0.0
        if len(self.recent_trades_pnl) > 0:
            avg_recent_pnl = np.mean(self.recent_trades_pnl[-5:]) if len(self.recent_trades_pnl) >= 5 else 0
            win_rate_recent = sum(1 for p in self.recent_trades_pnl[-5:] if p > 0) / min(5, len(self.recent_trades_pnl))
            trade_quality = avg_recent_pnl * 100 + win_rate_recent
        
        # Состояние позиции (12 параметров)
        position_state = np.array([
            self.position,
            min(1.0, self.steps_since_open / 100.0),
            min(1.0, self.steps_since_last_trade / 50.0),
            self.consecutive_losses / 5.0,
            self.consecutive_wins / 5.0,
            (self.net_worth - self.initial_balance) / self.initial_balance,
            min(1.0, max(0.0, (self.max_net_worth - self.net_worth) / max(self.max_net_worth, 1e-9))),
            min(1.0, self.active_margin / max(self.balance, 1e-9)),
            self.consecutive_profitable_trades / 10.0,
            self.consecutive_loss_trades / 10.0,
            trade_quality / 2.0,
            min(2.0, max(-1.0, self.avg_profit_last_10))
        ], dtype=np.float32)
        
        observation = np.concatenate([market_data, position_state])
        
        if np.any(np.isnan(observation)):
            observation = np.nan_to_num(observation, nan=0.0)
        
        return observation
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Шаг с усиленным фокусом на качество сделок"""
        if len(self.df) == 0:
            terminated = True
            truncated = True
            reward = 0.0
            return np.zeros(self.observation_space.shape[0], dtype=np.float32), reward, terminated, truncated, {}
        
        if self.current_step >= len(self.df) - 1:
            terminated = self.net_worth <= self.initial_balance * 0.3
            truncated = True
            reward = 0.0
            return self._get_observation(), reward, terminated, truncated, self._get_info()
        
        prev_net_worth = self.net_worth
        prev_position = self.position
        
        try:
            current_price = float(self.df.loc[self.current_step, "close"])
            if pd.isna(current_price):
                current_price = self.entry_price if self.entry_price > 0 else 100.0
        except:
            current_price = self.entry_price if self.entry_price > 0 else 100.0
        
        try:
            current_atr = float(self.df.loc[self.current_step, "atr"])
            if pd.isna(current_atr) or current_atr <= 0:
                current_atr = current_price * 0.01
        except:
            current_atr = current_price * 0.01
        
        self.current_step += 1
        self.num_timesteps += 1  # Увеличиваем счетчик шагов
        
        trade_opened = False
        trade_closed = False
        partial_close_occurred = False

        if self.current_step % 96 == 0:
            self.trades_today = 0
            self.current_day += 1

        # 1. Проверка закрытия позиции
        if self.position != 0:
            self.steps_since_open += 1
            
            partial_close_occurred = self._check_partial_tp(current_price, current_atr)
            
            if partial_close_occurred:
                print(f"[PARTIAL_CLOSE] Частичное закрытие на шаге {self.current_step}")
            
            if not partial_close_occurred:
                self._update_trailing_stop(current_price, current_atr)
            
            should_close_fully = self._check_full_close(current_price)
            
            if self.steps_since_open >= self.max_hold_steps and not should_close_fully:
                self.exit_type = "TIME_EXIT"
                self._close_position(current_price)
                trade_closed = True
                self.time_exit_count += 1
                print(f"[TIME_EXIT] Закрытие по времени")
            
            elif should_close_fully:
                print(f"📉 [TRADE] FULL CLOSE {'LONG' if self.position == 1 else 'SHORT'} at {self.actual_exit_price:.2f} (Reason: {self.exit_type}, Step {self.current_step})")
                self._close_position(self.actual_exit_price)
                trade_closed = True
            
            elif self._should_close_by_action(action, prev_position):
                if self.steps_since_open >= self.max_hold_steps * 0.8:
                    self.exit_type = "MANUAL"
                    print(f"📉 [TRADE] MANUAL CLOSE {'LONG' if self.position == 1 else 'SHORT'} at {current_price:.2f} (Step {self.current_step})")
                    self._close_position(current_price)
                    trade_closed = True
                    self.manual_count += 1
                    print(f"[MANUAL] Закрытие по действию")
        
        # 2. Открытие новой позиции - ТОЛЬКО ЕСЛИ ВСЕ ФИЛЬТРЫ ПРОЙДЕНЫ
        if not trade_closed and self.position == 0:
            if self.steps_since_last_trade >= self.trade_cooldown_steps:
                if self.trades_today < self.max_daily_trades:
                    
                    # V2: ЖЕСТКИЙ БАЛАНСИРОВЩИК - ПЕРЕКЛЮЧАЕМ action при перекосе
                    total_trades = self.long_trades_count + self.short_trades_count
                    if total_trades >= 3:  # После 3 сделок включаем балансировщик
                        # ПРИНУДИТЕЛЬНО переключаем на противоположный action при перекосе
                        if action == 1 and self.long_trades_count > self.short_trades_count * 1.5:
                            action = 2  # Принудительно SHORT вместо LONG
                            # Логи отключены: print(f"🔄 [BALANCE] LONG→SHORT: перекос {self.long_trades_count}/{self.short_trades_count}")
                        elif action == 2 and self.short_trades_count > self.long_trades_count * 1.5:
                            action = 1  # Принудительно LONG вместо SHORT
                            # Логи отключены: print(f"🔄 [BALANCE] SHORT→LONG: перекос {self.short_trades_count}/{self.long_trades_count}")
                    
                    # ЖЕСТКИЙ ФИЛЬТР ВХОДА С ГАРАНТИЕЙ RR
                    can_enter = self._check_entry_filters_strict(current_price, current_atr, action=action)
                    
                    if can_enter:
                        prev_pos_before_open = self.position
                        # Логи отключены для уменьшения спама
                        
                        if action == 1:  # Long
                            self._open_long_with_tp_features(current_price, current_atr)
                            print(f"🚀 [TRADE] OPEN LONG at {current_price:.2f} (Step {self.current_step}) | Balance: L={self.long_trades_count} S={self.short_trades_count}")
                            trade_opened = True
                            self.trades_today += 1
                        elif action == 2:  # Short
                            self._open_short_with_tp_features(current_price, current_atr)
                            print(f"🚀 [TRADE] OPEN SHORT at {current_price:.2f} (Step {self.current_step}) | Balance: L={self.long_trades_count} S={self.short_trades_count}")
                            trade_opened = True
                            self.trades_today += 1
                        # Логи отключены

        # 3. Обновление временных метрик и адаптивных фильтров
        if self.position == 0 and not trade_closed:
            self.steps_since_last_trade += 1
            self.steps_without_trade += 1
        else:
            # Если открыта или закрыта сделка, сбрасываем счетчик
            if trade_opened or trade_closed:
                self.steps_without_trade = 0
                self.last_trade_step = self.current_step
        
        # 3.1. Обновление адаптивных фильтров
        self._update_adaptive_filters()
        
        # 4. Обновление капитала
        self._update_net_worth(current_price)
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        self.net_worth_history.append(self.net_worth)
        
        # 5. Расчет награды с УСИЛЕННЫМ фокусом на качество
        reward = self._calculate_reward_profit_focused(
            prev_net_worth, 
            trade_opened, 
            trade_closed, 
            partial_close_occurred,
            current_price,
            action
        )
        self.rewards_history.append(reward)
        self.actions_history.append(action)
        
        # 6. Обновление статистики качества
        self._update_quality_stats(reward, trade_closed, partial_close_occurred)
        
        # 7. Проверка условий завершения
        terminated = self.net_worth <= self.initial_balance * 0.3
        truncated = self.current_step >= len(self.df) - 1
        
        info = self._get_info()
        
        return self._get_observation(), float(reward), terminated, truncated, info
    
    def _check_entry_filters_strict(self, price: float, atr: float, action: int = None) -> bool:
        """УСИЛЕННЫЕ ФИЛЬТРЫ НА ОСНОВЕ АНАЛИЗА ВАЖНОСТИ ПРИЗНАКОВ"""
        if self.current_step >= len(self.df):
            return False
        
        # ЭКСТРЕННЫЙ РЕЖИМ: если слишком долго нет сделок (>200 шагов = 2 дня), РАССЛАБЛЯЕМ пороги (не отключаем ключевые фильтры)
        emergency_mode = self.steps_without_trade > 200
        if emergency_mode and not hasattr(self, '_emergency_mode_logged'):
            print(f"⚠️ [EMERGENCY MODE] steps_without_trade={self.steps_without_trade} > 200 (≈2 дня). Расслабляем пороги фильтров.")
            self._emergency_mode_logged = True
        
        # Отладочная информация (только первые несколько раз)
        debug_filter = False
        if not hasattr(self, '_filter_debug_count'):
            self._filter_debug_count = 0
        if self._filter_debug_count < 5:
            debug_filter = True
            self._filter_debug_count += 1
        
        try:
            # 1) БАЗОВЫЙ ФИЛЬТР ATR
            atr_percent = atr / price
            if atr_percent < 0.0003 or atr_percent > 0.06:
                return False
            
            # 2) volatility_ratio (самый важный признак)
            if 'volatility_ratio' in self.df.columns:
                try:
                    vol_ratio = float(self.df.loc[self.current_step, 'volatility_ratio'])
                    if action == 1:  # LONG
                        lo, hi = (0.0020, 0.0075) if emergency_mode else (0.0025, 0.0060)
                        if vol_ratio < lo or vol_ratio > hi:
                            return False
                    elif action == 2:  # SHORT
                        # SHORT сейчас показывает плохой WR → делаем диапазон более “качественным”, но с послаблением в emergency
                        lo, hi = (0.0015, 0.0090) if emergency_mode else (0.0020, 0.0075)
                        if vol_ratio < lo or vol_ratio > hi:
                            return False
                except Exception:
                    pass
            
            # 3) Volume (разница WR ~50%)
            if 'volume' in self.df.columns:
                try:
                    volume = float(self.df.loc[self.current_step, 'volume'])
                    lookback = min(20, self.current_step)
                    if lookback > 0:
                        avg_volume = self.df.loc[max(0, self.current_step - lookback):self.current_step, 'volume'].mean()
                        volume_ratio = volume / avg_volume if avg_volume and avg_volume > 0 else 1.0
                        if action == 1:  # LONG - УЖЕСТОЧЕНО для качества
                            # 🔥 В emergency_mode ОСЛАБЛЯЕМ до минимума (0.9 = 90% среднего объема)
                            thr = 0.9 if emergency_mode else 1.40  # УВЕЛИЧЕНО с 1.30 до 1.40
                            if volume_ratio < thr:
                                return False
                        elif action == 2:  # SHORT
                            # SHORT плохо работает → не шортим на "пустом" объёме
                            thr = 0.70 if emergency_mode else 1.00  # ОСЛАБЛЕНО в emergency_mode
                            if volume_ratio < thr:
                                return False
                except Exception:
                    pass
            
            # 4) RSI (особенно важен для LONG) - УЖЕСТОЧЕНО ДЛЯ LONG
            rsi_norm_val = None
            if 'rsi_norm' in self.df.columns:
                try:
                    rsi_norm = float(self.df.loc[self.current_step, 'rsi_norm'])
                    rsi_norm_val = rsi_norm
                    if action == 1:
                        # 🔥 УЖЕСТОЧЕНО: более точная зона перепроданности для LONG
                        # Анализ показал: оптимальная зона для LONG - перепроданность, но не экстремальная
                        # В emergency_mode РАСШИРЯЕМ окно до (-0.7, 0.3) для возможности открытия сделок
                        lo, hi = (-0.7, 0.3) if emergency_mode else (-0.4, 0.0)  # УЖЕСТОЧЕНО: было (-0.3, 0.3)
                        if rsi_norm < lo or rsi_norm > hi:
                            return False
                    elif action == 2:
                        # SHORT: запрещаем шорт в перепроданности (главный источник плохих шортов)
                        lo, hi = (-0.2, 1.0) if emergency_mode else (0.10, 0.90)
                        if rsi_norm < lo or rsi_norm > hi:
                            return False
                except Exception:
                    pass
            
            # 5) ADX + DI - УЖЕСТОЧЕНО ДЛЯ LONG
            if 'adx' in self.df.columns:
                try:
                    adx_val = float(self.df.loc[self.current_step, 'adx'])
                    if action == 1:
                        # 🔥 LONG: требовать СИЛЬНОЕ преимущество +DI над -DI
                        if 'plus_di' in self.df.columns and 'minus_di' in self.df.columns:
                            plus_di = float(self.df.loc[self.current_step, 'plus_di'])
                            minus_di = float(self.df.loc[self.current_step, 'minus_di'])
                            # В emergency_mode ОСЛАБЛЯЕМ: просто plus_di >= minus_di (без требования 10% преимущества)
                            if emergency_mode:
                                if plus_di < minus_di * 0.95:  # В emergency_mode разрешаем даже если +DI немного меньше
                                    return False
                            else:
                                # УЖЕСТОЧЕНО: +DI должен быть минимум на 10% больше -DI
                                if plus_di <= minus_di * 1.10:  # БЫЛО: просто plus_di <= minus_di
                                    return False
                        # 🔥 ИСПОЛЬЗУЕМ АДАПТИВНЫЙ min_adx вместо жестко заданного значения!
                        min_adx = self.min_adx if emergency_mode else 25.0  # В emergency_mode используем адаптивный фильтр
                        if adx_val < min_adx:
                            return False
                    elif action == 2:
                        min_adx = 12.0 if emergency_mode else 15.0
                        if adx_val < min_adx:
                            return False
                        # SHORT: по умолчанию требуем нисходящее направление (minus_di > plus_di)
                        if 'plus_di' in self.df.columns and 'minus_di' in self.df.columns:
                            plus_di = float(self.df.loc[self.current_step, 'plus_di'])
                            minus_di = float(self.df.loc[self.current_step, 'minus_di'])
                            if minus_di <= plus_di:
                                # Разрешаем контр-тренд шорт ТОЛЬКО если явная перекупленность
                                if rsi_norm_val is None or rsi_norm_val < (0.40 if not emergency_mode else 0.20):
                                    return False
                except Exception:
                    pass
            
            # 6) trend_bias_1h (отрицательная корреляция — используем как фильтр “от противного”)
            if 'trend_bias_1h' in self.df.columns:
                try:
                    trend_bias = float(self.df.loc[self.current_step, 'trend_bias_1h'])
                    if action == 1 and trend_bias < -0.3:
                        return False
                    # SHORT: также не шортим при сильном бычьем уклоне
                    if action == 2 and trend_bias > (0.35 if emergency_mode else 0.25):
                        return False
                except Exception:
                    pass
            
            # 7) Контекст цены (доп. фильтр)
            if not emergency_mode:
                try:
                    if not self._check_price_context(price, action):
                        return False
                except Exception:
                    pass
            
            # 7.1) Anti-chasing по последней свече (быстрый фильтр) - УЖЕСТОЧЕНО ДЛЯ LONG
            try:
                if self.current_step > 0 and 'close' in self.df.columns:
                    prev_close = float(self.df.loc[self.current_step - 1, 'close'])
                    if prev_close > 0:
                        last_change_pct = (price - prev_close) / prev_close * 100.0
                        if action == 1:
                            # 🔥 УЖЕСТОЧЕНО: не покупаем после роста >1.0% (было 1.5%)
                            # Также используем ATR для определения "значимого" движения
                            if 'atr' in self.df.columns:
                                atr_val = float(self.df.loc[self.current_step, 'atr'])
                                atr_pct = (atr_val / prev_close) * 100.0
                                # Не покупаем если рост больше 1.5x ATR
                                if last_change_pct > max(1.0, atr_pct * 1.5):
                                    return False
                            elif last_change_pct > 1.0:  # Fallback если нет ATR
                                return False
                        if action == 2 and last_change_pct < -1.5:
                            return False  # не шортим после резкого падения
            except Exception:
                pass
            
            # 8) ГАРАНТИЯ MIN RR RATIO 1.5 (критично важно!)
            sl_distance = max(atr * self.atr_multiplier, price * self.min_sl_percent)
            sl_distance = min(sl_distance, price * self.max_sl_percent)
            min_tp_for_rr = sl_distance * self.min_rr_ratio
            min_tp_distance = max(min_tp_for_rr, atr * self.tp_levels[0], price * self.min_tp_percent)
            actual_rr = min_tp_distance / sl_distance if sl_distance > 0 else 0
            if actual_rr < self.min_rr_ratio - 0.01:
                self.min_rr_violations += 1
                return False
            
            # Сохраняем RR статистику
            self.rr_stats.append(actual_rr)
            if len(self.rr_stats) > 100:
                self.rr_stats.pop(0)
            
            return True
        
        except Exception:
            return False

    def _check_price_context(self, current_price: float, action: int) -> bool:
        """Проверка контекста цены для фильтрации входов (разница цены прибыльные/убыточные ~7.9%)"""
        if self.current_step < 20:
            return True
        try:
            lookback = 20
            start_idx = max(0, self.current_step - lookback)
            prev_data = self.df.loc[start_idx:self.current_step - 1]
            if len(prev_data) < 10:
                return True
            
            closes = prev_data['close'].values
            price_change = (closes[-1] - closes[0]) / closes[0] * 100
            
            highs = prev_data['high'].values if 'high' in prev_data.columns else None
            lows = prev_data['low'].values if 'low' in prev_data.columns else None
            if highs is not None and lows is not None and len(highs) > 0:
                avg_range = np.mean(highs - lows) / closes[0] * 100
            else:
                avg_range = 0.0
            
            sma_10 = np.mean(closes[-10:]) if len(closes) >= 10 else closes[-1]
            
            if action == 1:  # LONG - УЖЕСТОЧЕНО для качества
                # 🔥 Не покупаем в сильном нисходящем тренде
                if price_change < -2.5:  # УЖЕСТОЧЕНО: было -3.0
                    return False
                # 🔥 Не покупаем при слишком высокой волатильности (шум)
                if avg_range > 2.0:  # УЖЕСТОЧЕНО: было 2.5
                    return False
                # 🔥 Проверка расстояния от SMA с использованием ATR
                if 'atr' in self.df.columns:
                    try:
                        atr_val = float(self.df.loc[self.current_step, 'atr'])
                        atr_pct = (atr_val / sma_10) * 100.0
                        # Не покупаем если цена ниже SMA более чем на 2 ATR
                        price_deviation = (current_price - sma_10) / sma_10 * 100.0
                        if price_deviation < -2.0 * atr_pct:
                            return False
                    except:
                        # Fallback: не покупаем если цена ниже SMA на 1.5%
                        if current_price < sma_10 * 0.985:  # УЖЕСТОЧЕНО: было 0.98
                            return False
                else:
                    if current_price < sma_10 * 0.985:  # УЖЕСТОЧЕНО: было 0.98
                        return False
            elif action == 2:  # SHORT
                if price_change > 3.0:
                    return False
                # Не шортим после сильного падения (анти-chasing)
                if price_change < -3.0:
                    return False
                if avg_range < 0.8:
                    return False
                if current_price > sma_10 * 1.02:
                    return False
            
            return True
        except Exception:
            return True
    
    def _update_adaptive_filters(self):
        """Обновление адаптивных фильтров: ослабление при отсутствии сделок"""
        # Если долго нет сделок, ослабляем фильтры
        if self.steps_without_trade > self.max_steps_without_trade:
            # Ослабляем фильтры постепенно
            relaxation_factor = self.filter_relaxation_rate ** ((self.steps_without_trade - self.max_steps_without_trade) // 50)
            
            # Обновляем min_adx
            new_min_adx = max(
                self.min_filter_values['min_adx'],
                self.base_min_adx * relaxation_factor
            )
            self.min_adx = new_min_adx
            
            # Обновляем min_trend_strength
            new_min_trend_strength = max(
                self.min_filter_values['min_trend_strength'],
                self.base_min_trend_strength * relaxation_factor
            )
            self.min_trend_strength = new_min_trend_strength
            
            # Логируем каждые 100 шагов
            if self.steps_without_trade % 100 == 0:
                print(f"[ADAPTIVE_FILTERS] Ослабление фильтров: steps_without_trade={self.steps_without_trade}, "
                      f"min_adx={self.min_adx:.1f} (базовый {self.base_min_adx:.1f}), "
                      f"min_trend_strength={self.min_trend_strength:.2f} (базовый {self.base_min_trend_strength:.2f})")
        else:
            # Если сделки есть, постепенно возвращаем фильтры к базовым значениям
            if self.min_adx < self.base_min_adx or self.min_trend_strength < self.base_min_trend_strength:
                recovery_rate = 1.01  # Медленное восстановление (1% за шаг)
                self.min_adx = min(self.base_min_adx, self.min_adx * recovery_rate)
                self.min_trend_strength = min(self.base_min_trend_strength, self.min_trend_strength * recovery_rate)
    
    def _open_long_with_tp_features(self, price: float, atr: float):
        """Открытие лонг позиции с гарантированным RR"""
        self.entry_price = price * (1 + self.slippage)
        
        # Рассчитываем SL
        sl_distance = max(atr * self.atr_multiplier, price * self.min_sl_percent)
        sl_distance = min(sl_distance, price * self.max_sl_percent)
        self.initial_sl = self.entry_price - sl_distance
        self.current_sl = self.initial_sl
        
        # ГАРАНТИРУЕМ МИНИМАЛЬНЫЙ RR
        min_tp_for_rr = sl_distance * self.min_rr_ratio
        
        # Базовое расстояние для TP1
        base_tp_distance = max(
            min_tp_for_rr,
            atr * self.tp_levels[0],
            price * self.min_tp_percent
        )
        
        # Рассчитываем все TP уровни
        self.tp_prices = []
        for tp_mult in self.tp_levels:
            tp_distance = base_tp_distance * (tp_mult / self.tp_levels[0])
            tp_price = self.entry_price + tp_distance
            self.tp_prices.append(tp_price)
        
        # Проверяем RR (логи отключены)
        actual_rr = (self.tp_prices[0] - self.entry_price) / sl_distance if sl_distance > 0 else 0
        avg_rr = np.mean(self.rr_stats) if self.rr_stats else 0
        
        # Логи отключены для уменьшения спама
        # if actual_rr < self.min_rr_ratio:
        #     print(f"⚠️ [WARNING] RR низкий при открытии: {actual_rr:.2f}")
        
        self.tp_closed_levels = [False, False, False]
        self.partial_closes = []
        
        self._setup_trade_enhanced(position=1)

        # ЛОГИРУЕМ КАЧЕСТВО ВХОДА (без спама)
        try:
            row = self.df.iloc[self.current_step]
            entry_quality_score = 0
            if 'volatility_ratio' in row:
                vr = float(row['volatility_ratio'])
                if 0.003 < vr < 0.006:
                    entry_quality_score += 2
            if 'volume_ratio' in row:
                vrr = float(row['volume_ratio'])
                if vrr > 1.3:
                    entry_quality_score += 1
            if 'rsi_norm' in row:
                rsi_v = float(row['rsi_norm'])
                if -0.4 < rsi_v < 0.2:
                    entry_quality_score += 1
            if 'adx' in row:
                adx_v = float(row['adx'])
                if adx_v > 25:
                    entry_quality_score += 1
            if entry_quality_score >= 3:
                quality_level = "HIGH" if entry_quality_score >= 4 else "MEDIUM"
                print(f"[ENTRY_QUALITY] LONG {quality_level} ({entry_quality_score}/5)")
        except Exception:
            pass
    
    def _open_short_with_tp_features(self, price: float, atr: float):
        """Открытие шорт позиции с гарантированным RR"""
        self.entry_price = price * (1 - self.slippage)
        
        # Рассчитываем SL
        sl_distance = max(atr * self.atr_multiplier, price * self.min_sl_percent)
        sl_distance = min(sl_distance, price * self.max_sl_percent)
        self.initial_sl = self.entry_price + sl_distance
        self.current_sl = self.initial_sl
        
        # ГАРАНТИРУЕМ МИНИМАЛЬНЫЙ RR
        min_tp_for_rr = sl_distance * self.min_rr_ratio
        
        # Базовое расстояние для TP1
        base_tp_distance = max(
            min_tp_for_rr,
            atr * self.tp_levels[0],
            price * self.min_tp_percent
        )
        
        # Рассчитываем все TP уровни
        self.tp_prices = []
        for tp_mult in self.tp_levels:
            tp_distance = base_tp_distance * (tp_mult / self.tp_levels[0])
            tp_price = self.entry_price - tp_distance
            self.tp_prices.append(tp_price)
        
        # Проверяем RR (логи отключены)
        actual_rr = (self.entry_price - self.tp_prices[0]) / sl_distance if sl_distance > 0 else 0
        avg_rr = np.mean(self.rr_stats) if self.rr_stats else 0
        
        # Логи отключены для уменьшения спама
        # if actual_rr < self.min_rr_ratio:
        #     print(f"⚠️ [WARNING] RR низкий при открытии: {actual_rr:.2f}")
        
        self.tp_closed_levels = [False, False, False]
        self.partial_closes = []
        
        self._setup_trade_enhanced(position=-1)

        # ЛОГИРУЕМ КАЧЕСТВО ВХОДА (без спама)
        try:
            row = self.df.iloc[self.current_step]
            entry_quality_score = 0
            if 'volatility_ratio' in row:
                vr = float(row['volatility_ratio'])
                if 0.002 < vr < 0.008:
                    entry_quality_score += 2
            if 'volume_ratio' in row:
                vrr = float(row['volume_ratio'])
                if vrr > 1.0:
                    entry_quality_score += 1
            if 'adx' in row:
                adx_v = float(row['adx'])
                if adx_v > 15:
                    entry_quality_score += 1
            if 'trend_bias_1h' in row:
                tb = float(row['trend_bias_1h'])
                if tb < 0.3:
                    entry_quality_score += 1
            if entry_quality_score >= 3:
                quality_level = "HIGH" if entry_quality_score >= 4 else "MEDIUM"
                print(f"[ENTRY_QUALITY] SHORT {quality_level} ({entry_quality_score}/5)")
        except Exception:
            pass
    
    def _setup_trade_enhanced(self, position: int):
        """Настройка сделки с управлением рисками"""
        MIN_POSITION_SIZE = 0.01
        
        position_size = self.base_margin_percent
        
        # Динамическое управление размером позиции
        if self.consecutive_losses >= 2:
            position_size *= 0.5  # Уменьшаем при убытках
        elif self.consecutive_wins >= 2:
            position_size *= min(1.3, 1.0 + (self.consecutive_wins * 0.05))
        
        position_size = max(position_size, MIN_POSITION_SIZE)
        position_size = min(position_size, 0.1)  # Максимум 10%
        
        self.margin_percent = position_size
        self.active_margin = self.net_worth * position_size
        
        MIN_MARGIN = self.initial_balance * 0.01
        if self.active_margin < MIN_MARGIN:
            self.active_margin = MIN_MARGIN
        
        if self.active_margin > 0 and self.entry_price > 0:
            available_amount = self.active_margin * (1 - self.commission)
            MIN_SHARES = 0.001
            total_shares = max(available_amount / self.entry_price, MIN_SHARES)
            self.shares_held = total_shares
            self.shares_remaining = total_shares
        else:
            self.shares_held = 0
            self.shares_remaining = 0
        
        self.balance -= self.active_margin
        self.position = position
        self.steps_since_open = 0
        self.trailing_active = False
        self.exit_type = None
        self.highest_profit_pct = 0.0
        self.lowest_profit_pct = 0.0
        
        if position == -1:
            self.lowest_profit_pct = -0.0
    
    def _check_partial_tp(self, current_price: float, atr: float) -> bool:
        """Проверка частичного закрытия"""
        if self.position == 0 or all(self.tp_closed_levels):
            return False
        
        for i, tp_price in enumerate(self.tp_prices):
            if not self.tp_closed_levels[i]:
                should_close = False
                close_price = 0.0
                
                if self.position == 1:
                    if current_price >= tp_price:
                        should_close = True
                        close_price = max(tp_price, current_price * 0.999)
                else:
                    if current_price <= tp_price:
                        should_close = True
                        close_price = min(tp_price, current_price * 1.001)
                
                if should_close:
                    close_percentage = self.tp_close_percentages[i]
                    self._partial_close(close_percentage, close_price, i)
                    
                    if i == 0 and not self.trailing_active:
                        self.trailing_active = True
                        print(f"[TP{i+1}] Частичное закрытие, активация трейлинга")
                    else:
                        print(f"[TP{i+1}] Частичное закрытие")
                    
                    return True
        
        return False
    
    def _partial_close(self, percentage: float, price: float, tp_level: int):
        """Частичное закрытие позиции"""
        if self.shares_remaining <= 0 or self.shares_held <= 0:
            return
        
        shares_to_close = self.shares_held * percentage
        shares_to_close = min(shares_to_close, self.shares_remaining)
        
        if self.position == 1:
            pnl_ratio = (price - self.entry_price) / self.entry_price
            close_value = shares_to_close * price
            proceeds = close_value * (1 - self.commission)
        else:
            pnl_ratio = (self.entry_price - price) / self.entry_price
            margin_portion = (shares_to_close / self.shares_held) * self.active_margin
            proceeds = margin_portion * (1 + pnl_ratio) * (1 - self.commission)
        
        # Логирование
        trade_quality = "GOOD" if pnl_ratio > 0.01 else "NORMAL"
        # Расчет RR для частичного закрытия
        if self.position == 1:
            risk = self.entry_price - self.initial_sl
            reward = price - self.entry_price
            rr_ratio = reward / risk if risk > 0 else 0
        else:
            risk = self.initial_sl - self.entry_price
            reward = self.entry_price - price
            rr_ratio = reward / risk if risk > 0 else 0
        
        # Безопасное формирование строки TP уровней
        tp_str = ""
        if self.tp_prices and len(self.tp_prices) >= 3:
            tp_str = f"{self.tp_prices[0]:.4f},{self.tp_prices[1]:.4f},{self.tp_prices[2]:.4f}"
        elif self.tp_prices:
            tp_str = ",".join([f"{p:.4f}" for p in self.tp_prices])

        try:
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.current_step,
                    "LONG_PARTIAL" if self.position == 1 else "SHORT_PARTIAL",
                    round(self.entry_price, 4),
                    round(self.initial_sl, 4),
                    round(self.current_sl, 4),
                    tp_str,
                    round(price, 4),
                    f"{pnl_ratio*100:.2f}%",
                    round(self.net_worth, 2),
                    f"TP_LEVEL_{tp_level+1}",
                    self.steps_since_open,
                    "YES" if self.trailing_active else "NO",
                    str(tp_level+1),
                    len(self.partial_closes) + 1,
                    trade_quality,
                    f"{rr_ratio:.2f}"
                ])
                f.flush() # Принудительная запись на диск
        except Exception as e:
            print(f"Ошибка записи частичного закрытия: {e}")
        
        self.balance += proceeds
        self.shares_remaining -= shares_to_close
        
        if self.shares_remaining < 0:
            self.shares_remaining = 0
        
        if self.position == -1:
            self.active_margin -= margin_portion
            if self.active_margin < 0:
                self.active_margin = 0
        
        self.tp_closed_levels[tp_level] = True
        
        partial_close_info = {
            'tp_level': tp_level + 1,
            'percentage': percentage,
            'price': price,
            'pnl_ratio': pnl_ratio,
            'shares_closed': shares_to_close,
            'shares_remaining': self.shares_remaining,
            'step': self.current_step,
            'proceeds': proceeds,
            'rr_ratio': rr_ratio
        }
        self.partial_closes.append(partial_close_info)
        
        self.tp_count += 1
        
        # Обновление статистики качества
        if pnl_ratio > 0:
            self.consecutive_profitable_trades += 1
            self.consecutive_loss_trades = 0
        else:
            self.consecutive_loss_trades += 1
            self.consecutive_profitable_trades = 0
    
    def _update_trailing_stop(self, current_price: float, atr: float):
        """ПЕРЕПИСАННЫЙ ТРЕЙЛИНГ-СТОП - МЕНЕЕ АГРЕССИВНЫЙ"""
        if self.position == 0:
            return
        
        if self.position == 1:  # LONG
            profit_pct = (current_price - self.entry_price) / self.entry_price
            self.highest_profit_pct = max(self.highest_profit_pct, profit_pct)
            
            # Позже активируем трейлинг
            trailing_activation = 0.35
            if profit_pct >= (atr / self.entry_price) * trailing_activation:
                if not self.trailing_active:
                    self.trailing_active = True
                
                trailing_multiplier = self.trailing_distance_atr
                # Динамический трейлинг: больше прибыль = больше расстояние
                if profit_pct > 0.025:
                    trailing_multiplier *= 1.5
                elif profit_pct > 0.015:
                    trailing_multiplier *= 1.2
                
                trailing_stop_price = current_price - (atr * trailing_multiplier)
                # Обновляем только если новый SL лучше
                if trailing_stop_price > self.current_sl:
                    self.current_sl = trailing_stop_price
            
        else:  # SHORT
            profit_pct = (self.entry_price - current_price) / self.entry_price
            self.lowest_profit_pct = min(self.lowest_profit_pct, -profit_pct)
            
            trailing_distance = self.short_config.get('trailing_distance_atr', self.trailing_distance_atr)
            # Для SHORT активируем ещё позже
            trailing_activation = 0.45
            
            if profit_pct >= (atr / self.entry_price) * trailing_activation:
                if not self.trailing_active:
                    self.trailing_active = True
                
                trailing_multiplier = trailing_distance
                if profit_pct > 0.025:
                    trailing_multiplier *= 1.5
                elif profit_pct > 0.015:
                    trailing_multiplier *= 1.2
                
                trailing_stop_price = current_price + (atr * trailing_multiplier)
                # Обновляем только если новый SL лучше (ниже для SHORT)
                if trailing_stop_price < self.current_sl:
                    self.current_sl = trailing_stop_price
    
    def _check_full_close(self, current_price: float) -> bool:
        """Проверка полного закрытия"""
        if self.position == 0:
            return False
        
        if self.position == 1:
            if current_price <= self.current_sl:
                self.exit_type = "SL_TRAILING" if self.trailing_active else "SL_INITIAL"
                self.actual_exit_price = min(self.current_sl, current_price * 0.998)
                self.sl_count += 1
                return True
        else:
            if current_price >= self.current_sl:
                self.exit_type = "SL_TRAILING" if self.trailing_active else "SL_INITIAL"
                self.actual_exit_price = max(self.current_sl, current_price * 1.002)
                self.sl_count += 1
                return True
        
        if all(self.tp_closed_levels):
            self.exit_type = "TP_FULL"
            self.actual_exit_price = current_price
            self.tp_count += 1
            print(f"[TP_FULL] Все TP достигнуты")
            return True
        
        return False
    
    def _close_position(self, exec_price: float):
        """Метод закрытия позиции"""
        trade_type = "LONG" if self.position == 1 else "SHORT"
        total_pnl = 0.0
        
        # Расчет RR для сделки
        rr_ratio = 0.0
        if self.position == 1 and self.entry_price > 0 and self.initial_sl > 0:
            tp_achieved = self.tp_prices[0] if self.tp_prices else exec_price
            potential_profit = tp_achieved - self.entry_price
            risk = self.entry_price - self.initial_sl
            rr_ratio = potential_profit / risk if risk > 0 else 0
        elif self.position == -1 and self.entry_price > 0 and self.initial_sl > 0:
            tp_achieved = self.tp_prices[0] if self.tp_prices else exec_price
            potential_profit = self.entry_price - tp_achieved
            risk = self.initial_sl - self.entry_price
            rr_ratio = potential_profit / risk if risk > 0 else 0
        
        if self.position == 1:
            final_price = exec_price * (1 - self.slippage)
            
            partial_pnl = 0.0
            if self.partial_closes:
                for pc in self.partial_closes:
                    partial_pnl += pc['pnl_ratio'] * (pc['shares_closed'] / self.shares_held if self.shares_held > 0 else 0)
            
            if self.shares_remaining > 0:
                remaining_pnl = (final_price - self.entry_price) / self.entry_price
                remaining_weight = self.shares_remaining / self.shares_held if self.shares_held > 0 else 0
                remaining_pnl_weighted = remaining_pnl * remaining_weight
            else:
                remaining_pnl_weighted = 0
            
            total_pnl = partial_pnl + remaining_pnl_weighted
            
            if self.shares_remaining > 0:
                proceeds = self.shares_remaining * final_price * (1 - self.commission)
                self.balance += proceeds
        
        else:
            final_price = exec_price * (1 + self.slippage)
            total_pnl = (self.entry_price - final_price) / self.entry_price
            
            if self.active_margin > 0:
                margin_return = self.active_margin * (1 + total_pnl) * (1 - self.commission)
                self.balance += margin_return
        
        self.net_worth = self.balance
        
        # ОПРЕДЕЛЯЕМ КАЧЕСТВО СДЕЛКИ
        if total_pnl > 0.02:
            trade_quality = "EXCELLENT"
        elif total_pnl > 0.008:
            trade_quality = "GOOD"
        elif total_pnl > 0:
            trade_quality = "NORMAL"
        elif total_pnl > -0.005:
            trade_quality = "BAD"
        else:
            trade_quality = "VERY_BAD"
        
        # Логируем с RR
        pnl_percent = total_pnl * 100
        self._log_trade(final_price, pnl_percent, trade_type, trade_quality, rr_ratio)
        
        # Обновление статистики
        self.total_trades += 1
        self.total_pnl += total_pnl
        
        self.recent_trades_pnl.append(total_pnl)
        if len(self.recent_trades_pnl) > self.max_recent_trades:
            self.recent_trades_pnl.pop(0)
        
        if total_pnl > 0:
            self.winning_trades += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        
        trade_info = {
            'step': self.current_step,
            'type': trade_type,
            'entry_price': self.entry_price,
            'sl_price': self.current_sl,
            'tp_prices': self.tp_prices.copy(),
            'exit_price': final_price,
            'pnl': total_pnl,
            'exit_type': self.exit_type,
            'trailing_active': self.trailing_active,
            'duration': self.steps_since_open,
            'balance': self.balance,
            'net_worth': self.net_worth,
            'partial_closes': self.partial_closes.copy(),
            'tp_closed_levels': self.tp_closed_levels.copy(),
            'quality': trade_quality,
            'rr_ratio': rr_ratio
        }
        self.trade_history.append(trade_info)
        
        # Сброс позиции
        self.position = 0
        self.shares_held = 0
        self.shares_remaining = 0
        self.active_margin = 0
        self.tp_prices = []
        self.partial_closes = []
        self.tp_closed_levels = [False, False, False]
        self.trailing_active = False
    
    def _log_trade(self, exit_price: float, pnl_pct: float, trade_type: str, 
                  trade_quality: str = "NORMAL", rr_ratio: float = 0.0):
        """Логирование с RR ratio"""
        reason = self.exit_type if self.exit_type else "UNKNOWN"
        
        tp_closed_info = ""
        if self.tp_closed_levels:
            tp_closed = [i+1 for i, closed in enumerate(self.tp_closed_levels) if closed]
            tp_closed_info = ",".join(map(str, tp_closed)) if tp_closed else "NONE"
            
            if tp_closed and reason not in ["TP_FULL", "TP_PARTIAL"]:
                if len(tp_closed) < 3:
                    reason = "TP_PARTIAL"
                else:
                    reason = "TP_FULL"
        
        # Безопасное формирование строки TP уровней
        tp_str = ""
        if self.tp_prices and len(self.tp_prices) >= 3:
            tp_str = f"{self.tp_prices[0]:.4f},{self.tp_prices[1]:.4f},{self.tp_prices[2]:.4f}"
        elif self.tp_prices:
            tp_str = ",".join([f"{p:.4f}" for p in self.tp_prices])

        try:
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.current_step,
                    trade_type,
                    round(self.entry_price, 4),
                    round(self.initial_sl, 4),
                    round(self.current_sl, 4),
                    tp_str,
                    round(exit_price, 4),
                    f"{pnl_pct:.2f}%",
                    round(self.net_worth, 2),
                    reason,
                    self.steps_since_open,
                    "YES" if self.trailing_active else "NO",
                    tp_closed_info,
                    len(self.partial_closes),
                    trade_quality,
                    f"{rr_ratio:.2f}"
                ])
                f.flush() # Принудительная запись на диск
        except Exception as e:
            print(f"Ошибка записи в лог: {e}")
    
    def _should_close_by_action(self, action: int, prev_position: int) -> bool:
        """Проверка закрытия по действию"""
        if self.steps_since_open < self.min_hold_steps:
            return False
        
        if action == 1 and prev_position == -1:
            return self.steps_since_open >= self.min_hold_steps * 2
        
        if action == 2 and prev_position == 1:
            return self.steps_since_open >= self.min_hold_steps * 2
        
        return False
    
    def _update_net_worth(self, current_price: float):
        """Обновление чистой стоимости"""
        if self.position == 1 and self.shares_remaining > 0:
            current_value = self.shares_remaining * current_price
            self.net_worth = self.balance + current_value
        elif self.position == -1 and self.active_margin > 0:
            pnl_ratio = (self.entry_price - current_price) / self.entry_price
            current_margin_value = self.active_margin * (1 + pnl_ratio)
            self.net_worth = self.balance + current_margin_value
        else:
            self.net_worth = self.balance
    
    def _calculate_reward_profit_focused(self, prev_net_worth: float, 
                                      trade_opened: bool, 
                                      trade_closed: bool,
                                      partial_close: bool,
                                      current_price: float,
                                      action: int) -> float:
        """Улучшенная reward функция с акцентом на RR"""
        reward = 0.0
        
        # Базовый reward за изменение капитала
        equity_change = (self.net_worth - prev_net_worth) / self.initial_balance
        reward += np.tanh(equity_change * 40.0) * 1.5
        
        # БОНУС ЗА ОТКРЫТИЕ ПОЗИЦИИ В ХОРОШИХ УСЛОВИЯХ (КРИТИЧНО ДЛЯ АКТИВНОСТИ)
        if trade_opened:
            try:
                row = self.df.iloc[self.current_step]
                # ЗАМЕНА trend_bias_1h на ADX (сила тренда) и +DI/-DI (направление)
                adx_value = row.get('adx', 0)
                plus_di = row.get('plus_di', 0)
                minus_di = row.get('minus_di', 0)
                volume_ratio = row.get('volume_ratio', 1.0)
                volatility_ratio = row.get('volatility_ratio', 1.5)
                # ВАЖНО: rsi_norm должен быть СО ЗНАКОМ.
                # abs(...) ломает логику (LONG никогда не попадает в отрицательный диапазон).
                rsi_norm = float(row.get('rsi_norm', 0))
                prev_long = self.long_trades_count
                prev_short = self.short_trades_count
                
                # V2 ULTRA-SIMPLIFIED REWARD: УБИРАЕМ ВСЕ ПРОВЕРКИ ADX/DI/RSI
                # Модель сама должна научиться через PnL, когда открывать LONG/SHORT
                
                # ТОЛЬКО базовый бонус за открытие + баланс
                if self.position == 1:
                    reward += 2.0  # Базовый бонус за LONG
                    self.long_trades_count += 1
                elif self.position == -1:
                    reward += 2.0  # Равный бонус за SHORT
                    self.short_trades_count += 1
                
                # УСИЛЕННЫЙ БАЛАНСИРОВЩИК LONG/SHORT
                total_prev = prev_long + prev_short
                if total_prev >= 5:  # Раньше срабатывает (было 10)
                    if self.position == 1 and prev_long > prev_short * 1.2:  # Ужесточено (было 1.5)
                        reward -= 3.0  # Усилено (было 1.0)
                    if self.position == -1 and prev_short > prev_long * 1.2:
                        reward -= 3.0
                    if self.position == 1 and prev_long < prev_short * 0.8:  # Ослаблено (было 0.7)
                        reward += 5.0  # Усилено (было 2.0)
                    if self.position == -1 and prev_short < prev_long * 0.8:
                        reward += 5.0  # Усилено (было 2.0)
                
                # ВСЕ ПРОВЕРКИ ТРЕНДА УБРАНЫ - модель учится через PnL
                
                # Минимальный бонус за волатильность (чтобы не открывать в флэте)
                if volume_ratio > 1.0 and volatility_ratio >= self.min_volatility_ratio and volatility_ratio < 2.0:
                    reward += 1.0
                elif volume_ratio > 1.0:
                    reward += 0.5  # Минимальный бонус
            except (IndexError, KeyError):
                pass
        
        # V2: ШТРАФ ЗА ПРОПУСК УБРАН - модель сама решает через PnL
        # (Слишком агрессивный штраф за HOLD мешал модели учиться)
        
        # УЛУЧШЕННАЯ НАГРАДА ЗА TP С ХОРОШИМ RR (более сбалансированная)
        if partial_close:
            if self.partial_closes:
                last_close = self.partial_closes[-1]
                tp_level = last_close['tp_level']
                pnl_ratio = last_close['pnl_ratio']
                rr_ratio = last_close.get('rr_ratio', 1.5)
                
                # Базовая награда за достижение TP (пропорциональна уровню)
                tp_bonus = 15.0 * (tp_level * 0.85)  # Сбалансированная награда
                
                # Бонус за PnL (логарифмический для избежания экстремальных значений)
                pnl_bonus = min(25.0, np.log1p(abs(pnl_ratio) * 100) * 5.0)  # Логарифмический бонус
                
                # ДОПОЛНИТЕЛЬНЫЙ БОНУС ЗА ХОРОШИЙ RR (новое!)
                if rr_ratio >= 2.0:
                    rr_bonus = (rr_ratio - 1.5) * 3.0  # Бонус за RR выше 1.5
                    reward += rr_bonus
                
                reward += tp_bonus + pnl_bonus
                # Логи отключены
        
        # ОГРОМНЫЙ ШТРАФ ЗА SL С ПЛОХИМ RR
        elif trade_closed and self.exit_type in ["SL_INITIAL", "SL_TRAILING"]:
            pnl_sl = (self.net_worth / prev_net_worth) - 1
            
            # ШТРАФ ЗА ТРЕЙЛИНГ-SL ЗАКРЫТИЯ (если слишком частые)
            if self.exit_type == "SL_TRAILING":
                self.trailing_sl_count += 1
                self.recent_trailing_sl.append(1)
            else:
                # Для не-трейлинг-SL закрытий добавляем 0
                self.recent_trailing_sl.append(0)
            
            # Ограничиваем размер истории
            if len(self.recent_trailing_sl) > 10:
                self.recent_trailing_sl.pop(0)
            
            # Штраф за слишком частые трейлинг-SL закрытия
            if len(self.recent_trailing_sl) >= 5:
                trailing_sl_ratio = sum(self.recent_trailing_sl) / len(self.recent_trailing_sl)
                if trailing_sl_ratio > 0.5:  # Если >50% последних закрытий - трейлинг-SL
                    reward -= trailing_sl_ratio * 0.2  # Штраф пропорционален частоте
            
            # Проверяем RR сделки
            if self.trade_history:
                last_trade = self.trade_history[-1]
                rr = last_trade.get('rr_ratio', 1.0)
                
                # НАГРАДА ЗА ВЫСОКИЙ RR (>2.0)
                if rr > 2.0:
                    reward += rr * 0.1  # Бонус за высокий RR
                
                if rr < 1.0:
                    # Дополнительный штраф за плохой RR
                    reward -= 3.0 * (1.0 - rr)
                    # Логи отключены
            
            # УЛУЧШЕННЫЕ штрафы за SL (усилены)
            if pnl_sl < -0.02:
                reward -= 12.0  # УВЕЛИЧЕНО с 8.0 до 12.0
                # Логи отключены
            elif pnl_sl < -0.01:
                reward -= 8.0  # УВЕЛИЧЕНО с 4.0 до 8.0
                # Логи отключены
            else:
                reward -= 6.0  # УВЕЛИЧЕНО с 4.0 до 6.0
                # Логи отключены
        
        # ШТРАФ ЗА VERY_BAD СДЕЛКИ (дополнительно к штрафу за убыток)
        if trade_closed and self.trade_history:
            last_trade = self.trade_history[-1]
            trade_quality = last_trade.get('trade_quality', 'NORMAL')
            if trade_quality == 'VERY_BAD':
                reward -= 5.0  # Дополнительный штраф за VERY_BAD сделки
                # Логи отключены
        
        # УЛУЧШЕННЫЙ БОНУС ЗА ХОРОШИЙ RR В ПОСЛЕДНИХ СДЕЛКАХ
        if len(self.rr_stats) >= 5:
            avg_recent_rr = np.mean(self.rr_stats[-5:])
            if avg_recent_rr > 2.0:
                rr_bonus = min(10.0, (avg_recent_rr - 2.0) * 3.0)  # УВЕЛИЧЕНО с 6.0 до 10.0
                reward += rr_bonus
                # Логи отключены
            elif avg_recent_rr < 1.5:
                # Штраф за плохой средний RR
                rr_penalty = (1.5 - avg_recent_rr) * 2.0
                reward -= rr_penalty
                # Логи отключены
        
        # БОНУС ЗА СЕРИЮ ПРИБЫЛЬНЫХ СДЕЛОК (улучшено)
        if trade_closed and len(self.recent_trades_pnl) > 0:
            last_pnl = self.recent_trades_pnl[-1]
            if last_pnl > 0:
                self.win_streak += 1
                # Бонус за серию прибыльных сделок (пропорционален длине серии)
                reward += self.win_streak * 0.05  # Бонус за серию
            else:
                self.win_streak = 0  # Сбрасываем серию при убытке
        
        # БОНУС ЗА ПОСЛЕДОВАТЕЛЬНЫЕ ПРИБЫЛИ (дополнительный)
        if len(self.recent_trades_pnl) >= 3:
            recent_profits = [p for p in self.recent_trades_pnl[-3:] if p > 0]
            if len(recent_profits) == 3:
                consecutive_bonus = 5.0  # Бонус за 3 прибыли подряд
                reward += consecutive_bonus
                # Логи отключены
        
        # ШТРАФ ЗА ВСЕГДА ВЫБИРАТЬ ОДНО ДЕЙСТВИЕ (КРИТИЧНО ДЛЯ РАЗНООБРАЗИЯ)
        if len(self.actions_history) >= 50:
            recent_actions = self.actions_history[-50:]
            action_counts = {}
            for a in recent_actions:
                action_counts[a] = action_counts.get(a, 0) + 1
            
            # Если одно действие выбирается > 80% времени - штраф
            max_action_ratio = max(action_counts.values()) / len(recent_actions)
            if max_action_ratio > 0.8:
                diversity_penalty = (max_action_ratio - 0.8) * 10.0  # Штраф до 2.0
                reward -= diversity_penalty
                # Логи отключены
        
        # БОНУС ЗА РАЗНООБРАЗИЕ СДЕЛОК (НОВОЕ!)
        # Поощряем баланс между LONG и SHORT сделками
        total_trades = self.long_trades_count + self.short_trades_count
        if total_trades >= 10:
            long_ratio = self.long_trades_count / total_trades
            short_ratio = self.short_trades_count / total_trades
            # Бонус за баланс: если соотношение между 0.3 и 0.7 для каждого типа
            if 0.3 <= long_ratio <= 0.7 and 0.3 <= short_ratio <= 0.7:
                balance_ratio = min(long_ratio, short_ratio) / max(long_ratio, short_ratio)
                diversity_bonus = balance_ratio * 0.1  # Бонус за баланс (до 0.1)
                reward += diversity_bonus
                # Логи отключены: бонус за разнообразие сделок
        
        return np.clip(reward, -15.0, 35.0)  # Расширен диапазон для больших наград/штрафов
    
    def _update_quality_stats(self, reward: float, trade_closed: bool, partial_close: bool):
        """Обновление статистики качества"""
        if trade_closed and self.total_trades > 0:
            if len(self.recent_trades_pnl) > 0:
                self.avg_profit_last_10 = np.mean(self.recent_trades_pnl[-min(10, len(self.recent_trades_pnl))])
    
    def _get_info(self) -> Dict:
        """Получение информации о состоянии с RR статистикой"""
        win_rate = self.winning_trades / max(1, self.total_trades)
        
        avg_win = 0
        if self.winning_trades > 0:
            winning_pnls = [t['pnl'] for t in self.trade_history if t['pnl'] > 0]
            avg_win = np.mean(winning_pnls) if winning_pnls else 0
        
        avg_loss = 0
        if self.losing_trades > 0:
            losing_pnls = [t['pnl'] for t in self.trade_history if t['pnl'] < 0]
            avg_loss = abs(np.mean(losing_pnls)) if losing_pnls else 0
        
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        
        drawdown = 0
        if self.max_net_worth > 0:
            drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
        
        # Статистика качества сделок
        quality_stats = {
            'EXCELLENT': len([t for t in self.trade_history if t.get('quality') == 'EXCELLENT']),
            'GOOD': len([t for t in self.trade_history if t.get('quality') == 'GOOD']),
            'NORMAL': len([t for t in self.trade_history if t.get('quality') == 'NORMAL']),
            'BAD': len([t for t in self.trade_history if t.get('quality') == 'BAD']),
            'VERY_BAD': len([t for t in self.trade_history if t.get('quality') == 'VERY_BAD'])
        }
        
        # RR статистика
        rr_stats = {
            'min': min(self.rr_stats) if self.rr_stats else 0,
            'max': max(self.rr_stats) if self.rr_stats else 0,
            'avg': np.mean(self.rr_stats) if self.rr_stats else 0,
            'median': np.median(self.rr_stats) if self.rr_stats else 0,
            'violations': self.min_rr_violations,
            'count': len(self.rr_stats)
        }
        
        return {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'position': self.position,
            'current_step': self.current_step,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'quality_stats': quality_stats,
            'rr_stats': rr_stats,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_pnl': self.total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'consecutive_losses': self.consecutive_losses,
            'consecutive_wins': self.consecutive_wins,
            'max_net_worth': self.max_net_worth,
            'drawdown': drawdown,
            'trades_today': self.trades_today,
            'current_day': self.current_day,
            'recent_profit_factor': self.avg_profit_last_10 * 10,
            'consecutive_profitable': self.consecutive_profitable_trades,
            'consecutive_loss': self.consecutive_loss_trades
        }
    
    def render(self, mode='human'):
        """Визуализация текущего состояния"""
        if mode == 'human':
            print(f"Step: {self.current_step}, Net Worth: ${self.net_worth:.2f}, "
                  f"Position: {self.position}, Trades Today: {self.trades_today}")