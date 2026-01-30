"""
МУЛЬТИТАЙМФРЕЙМОВАЯ ВЕРСИЯ СРЕДЫ ОБУЧЕНИЯ V18 MTF (НЕЗАВИСИМАЯ)
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
import matplotlib.pyplot as plt
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
    MTF_MIN_1H_ADX,
    MTF_MIN_1H_ADX_SHORT,
    MTF_DI_RATIO_1H,
    MTF_TREND_CONFLICT_MULTIPLIER,
    MTF_REQUIRE_4H_CONFIRMATION,
    MTF_OPTIMIZED_PARAMS,
)


class CryptoTradingEnvV18_MTF(gym.Env):
    """
    МУЛЬТИТАЙМФРЕЙМОВАЯ ВЕРСИЯ V18 (НЕЗАВИСИМАЯ)
    Самостоятельный класс, объединяющий логику V17_2_Optimized и MTF расширения.
    """
    
    metadata = {'render.modes': ['human']}
    
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
        """
        super(CryptoTradingEnvV18_MTF, self).__init__()
        
        # 1. MTF данные
        self.df_15m = df_list[0] if len(df_list) > 0 else pd.DataFrame()
        self.df = self._prepare_data_simple(self.df_15m.copy())
        self.df_1h = df_list[1] if len(df_list) > 1 else None
        self.df_4h = df_list[2] if len(df_list) > 2 else None
        self.df_1d = df_list[3] if len(df_list) > 3 else None
        
        self.obs_cols = obs_cols
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.log_file = log_file
        self.log_open_positions = bool(log_open_positions)
        self.open_log_file = open_log_file
        
        # 2. Базовые параметры (из V17_2_Optimized)
        self.base_rr_ratio = rr_ratio
        self.atr_multiplier = atr_multiplier
        self.min_rr_ratio = 1.5
        self.tp_close_percentages = [0.25, 0.35, 0.40]
        self.protective_trailing_atr = 0.5
        self.max_hold_steps = 60
        self.min_hold_steps = 8
        self.min_sl_percent = 0.003
        self.max_sl_percent = 0.007
        self.min_tp_percent = 0.006
        self.base_margin_percent = 0.07
        self.max_daily_trades = 5
        self.min_trend_strength = 0.55
        self.min_volume_multiplier = 1.4
        self.min_price_distance_pct = 1.0
        self.min_price_distance_pct_short = 0.5
        
        # 3. MTF параметры и оптимизации V18
        self.mtf_enabled = self.df_1h is not None and self.df_4h is not None
        self.mtf_strict_mode = True
        self._mtf_index_cache = {}
        
        # Применение оптимизированных параметров V18
        self.min_volatility_ratio = MTF_MIN_VOLATILITY_RATIO
        self.max_volatility_ratio = MTF_MAX_VOLATILITY_RATIO
        self.trailing_activation_atr = MTF_TRAILING_ACTIVATION_ATR
        self.trailing_distance_atr = MTF_TRAILING_DISTANCE_ATR
        self.tp_levels = MTF_TP_LEVELS
        
        self.long_config = {
            'min_trend_strength': 0.50,
            'min_rsi_norm': MTF_LONG_RSI_MIN,
            'max_rsi_norm': MTF_LONG_RSI_MAX,
            'trailing_distance_atr': 0.35,
            'position_size_multiplier': 1.0,
        }
        
        self.short_config = {
            'min_trend_strength': 0.55,
            'min_rsi_norm': MTF_SHORT_RSI_MIN,
            'max_rsi_norm': MTF_SHORT_RSI_MAX,
            'trailing_distance_atr': 0.40,
            'position_size_multiplier': 0.7,
        }
        
        self.min_adx = MTF_MIN_ADX
        self.mtf_min_adx_short = MTF_MIN_ADX_SHORT
        self.mtf_min_absolute_atr = MTF_MIN_ABSOLUTE_ATR
        self.mtf_atr_percent_min = MTF_ATR_PERCENT_MIN
        self.mtf_min_absolute_volume = MTF_MIN_ABSOLUTE_VOLUME
        self.mtf_min_volume_spike = MTF_MIN_VOLUME_SPIKE
        self.mtf_min_volume_spike_short = MTF_MIN_VOLUME_SPIKE_SHORT
        self.mtf_min_1h_adx = MTF_MIN_1H_ADX
        self.mtf_min_1h_adx_short = MTF_MIN_1H_ADX_SHORT
        self.mtf_di_ratio_1h = MTF_DI_RATIO_1H
        self.mtf_trend_conflict_multiplier = MTF_TREND_CONFLICT_MULTIPLIER
        self.mtf_require_4h_confirmation = MTF_REQUIRE_4H_CONFIRMATION
        
        # Масштабирование признаков
        self.obs_scaling = {
            'rsi_norm': 3.0,
            'volume': 2.0,
            'atr': 1.5,
            'close': 1.0,
            'volume_ratio': 0.5,
        }
        
        # Reward параметры
        self.tp_bonus_multiplier = 15.0
        self.tp_full_bonus = 20.0
        self.sl_penalty_multiplier = 8.0
        self.manual_penalty = 4.0
        self.time_exit_penalty = 2.0
        self.quality_bonus_threshold = 0.015
        self.quality_bonus = 12.0
        
        # 4. Пространство действий и наблюдений
        self.action_space = spaces.Discrete(3)
        
        mtf_features_count = self._count_mtf_features()
        base_features_count = len(obs_cols) + 12
        n_features = base_features_count + mtf_features_count
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features,),
            dtype=np.float32
        )
        
        # 5. Инициализация состояния и логов
        self.recent_trades_pnl = []
        self.max_recent_trades = 20
        
        log_dir = os.path.dirname(log_file) if os.path.dirname(log_file) else "."
        if log_dir != ".":
            os.makedirs(log_dir, exist_ok=True)
            
        if self.log_open_positions:
            if not self.open_log_file:
                base_dir = os.path.dirname(self.log_file) if os.path.dirname(self.log_file) else "."
                base_name = os.path.basename(self.log_file)
                self.open_log_file = os.path.join(base_dir, f"opens_{base_name}")
            self._init_open_log_file()
            
        self._init_log_file()
        
        # Статистика и история
        self.rr_stats = []
        self.min_rr_violations = 0
        self.recent_trailing_sl = []
        self.trade_history = []
        self.net_worth_history = []
        self.actions_history = []
        self.rewards_history = []
        
        print(f"[MTF_INIT] Размеры: базовые={base_features_count}, MTF={mtf_features_count}, всего={n_features}")
        if self.df_1h is not None:
            self._validate_mtf_sync()
            
        self.reset()

    # --- Методы из V17_2_Optimized ---
    
    def _prepare_data_simple(self, df: pd.DataFrame) -> pd.DataFrame:
        """Упрощенная подготовка данных"""
        if len(df) == 0:
            return pd.DataFrame()
        
        df = df.reset_index(drop=True).copy()
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 100.0
        
        if 'atr' not in df.columns:
            high_low = df['high'] - df['low']
            df['atr'] = high_low.rolling(window=14, min_periods=1).mean()
            df['atr'] = df['atr'].fillna(df['atr'].mean() if not df['atr'].isnull().all() else 1.0)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                mean_val = df[col].mean() if not df[col].isnull().all() else 0
                df[col] = df[col].fillna(mean_val)
        
        return df

    def _init_log_file(self):
        """Инициализация файла логов"""
        try:
            if os.path.exists(self.log_file):
                return
            with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'step', 'type', 'entry', 'sl_initial', 'sl_current',
                    'tp_levels', 'exit', 'pnl_percent', 'net_worth',
                    'exit_reason', 'duration', 'trailing', 'tp_closed', 'partial_closes',
                    'trade_quality', 'rr_ratio'
                ])
        except Exception as e:
            print(f"⚠️ Ошибка создания файла логов: {e}")

    def _init_open_log_file(self):
        """Инициализация файла логов открытия позиций"""
        if not self.open_log_file:
            return
        try:
            if os.path.exists(self.open_log_file):
                return
            with open(self.open_log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'step', 'side', 'entry', 'sl', 'tp1', 'tp2', 'tp3',
                    'rr_tp1', 'atr', 'atr_pct', 'margin_pct', 'active_margin', 'shares'
                ])
        except Exception as e:
            print(f"⚠️ Ошибка создания open-log файла: {e}")

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Сброс среды"""
        if seed is not None:
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
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.tp_count = 0
        self.sl_count = 0
        self.time_exit_count = 0
        self.manual_count = 0
        self.consecutive_profitable_trades = 0
        self.consecutive_loss_trades = 0
        self.avg_profit_last_10 = 0
        self.trailing_sl_count = 0
        self.win_streak = 0
        self.num_timesteps = 0
        self._mtf_index_cache = {}
        
        self.trade_history = []
        self.net_worth_history = [self.net_worth]
        self.actions_history = []
        self.rewards_history = []
        
        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Шаг среды"""
        if len(self.df) == 0 or self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0.0, False, True, self._get_info()
        
        prev_net_worth = self.net_worth
        prev_position = self.position
        
        current_price = float(self.df.loc[self.current_step, "close"])
        current_atr = float(self.df.loc[self.current_step, "atr"])
        
        self.current_step += 1
        self.num_timesteps += 1
        
        if self.current_step % 96 == 0:
            self.trades_today = 0
            self.current_day += 1
            
        trade_opened = False
        trade_closed = False
        partial_close_occurred = False
        closed_by_action = False

        # 1. Проверка закрытия
        if self.position != 0:
            self.steps_since_open += 1
            partial_close_occurred = self._check_partial_tp(current_price, current_atr)
            
            if not partial_close_occurred:
                self._update_trailing_stop(current_price, current_atr)
            
            should_close_fully = self._check_full_close(current_price)
            
            if self.steps_since_open >= self.max_hold_steps and not should_close_fully:
                self.exit_type = "TIME_EXIT"
                self._close_position(current_price)
                trade_closed = True
                self.time_exit_count += 1
            elif should_close_fully:
                self._close_position(self.actual_exit_price)
                trade_closed = True
            elif self._should_close_by_action(action, prev_position):
                if self.steps_since_open >= self.max_hold_steps * 0.8:
                    self.exit_type = "MANUAL"
                    self._close_position(current_price)
                    trade_closed = True
                    closed_by_action = True
                    self.manual_count += 1

        # 2. Открытие новой позиции
        if self.position == 0:
            allow_new_trade = self.steps_since_last_trade >= 10 or closed_by_action
            if allow_new_trade and (not trade_closed or closed_by_action):
                if self.trades_today < self.max_daily_trades:
                    force_trade = self.steps_since_last_trade >= 100
                    if force_trade or self._check_entry_filters_strict(current_price, current_atr, action=action):
                        if action == 1:
                            self._open_long_with_tp_features(current_price, current_atr)
                            trade_opened = True
                            self.trades_today += 1
                        elif action == 2:
                            self._open_short_with_tp_features(current_price, current_atr)
                            trade_opened = True
                            self.trades_today += 1

        if self.position == 0 and not trade_closed:
            self.steps_since_last_trade += 1
            
        self._update_net_worth(current_price)
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        
        reward = self._calculate_reward_profit_focused(
            prev_net_worth, trade_opened, trade_closed, partial_close_occurred, current_price, action
        )
        
        self._update_quality_stats(reward, trade_closed, partial_close_occurred)
        
        terminated = self.net_worth <= self.initial_balance * 0.3
        truncated = self.current_step >= len(self.df) - 1
        
        return self._get_observation(), float(reward), terminated, truncated, self._get_info()

    def _setup_trade_enhanced(self, position: int):
        """Настройка сделки"""
        position_size = self.base_margin_percent
        if self.consecutive_losses >= 2:
            position_size *= 0.5
        elif self.consecutive_wins >= 2:
            position_size *= min(1.3, 1.0 + (self.consecutive_wins * 0.05))
        
        position_size = np.clip(position_size, 0.01, 0.1)
        self.margin_percent = position_size
        self.active_margin = self.net_worth * position_size
        
        if self.active_margin < self.initial_balance * 0.01:
            self.active_margin = self.initial_balance * 0.01
            
        if self.active_margin > 0 and self.entry_price > 0:
            available_amount = self.active_margin * (1 - self.commission)
            self.shares_held = max(available_amount / self.entry_price, 0.001)
            self.shares_remaining = self.shares_held
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
                if self.position == 1 and current_price >= tp_price:
                    should_close = True
                    close_price = max(tp_price, current_price * 0.999)
                elif self.position == -1 and current_price <= tp_price:
                    should_close = True
                    close_price = min(tp_price, current_price * 1.001)
                
                if should_close:
                    self._partial_close(self.tp_close_percentages[i], close_price, i)
                    if i == 0:
                        self.trailing_active = True
                    return True
        return False

    def _partial_close(self, percentage: float, price: float, tp_level: int):
        """Частичное закрытие"""
        if self.shares_remaining <= 0:
            return
        
        shares_to_close = min(self.shares_held * percentage, self.shares_remaining)
        if self.position == 1:
            pnl_ratio = (price - self.entry_price) / self.entry_price
            proceeds = (shares_to_close * price) * (1 - self.commission)
            risk = self.entry_price - self.initial_sl
            reward = price - self.entry_price
        else:
            pnl_ratio = (self.entry_price - price) / self.entry_price
            margin_portion = (shares_to_close / self.shares_held) * self.active_margin
            proceeds = margin_portion * (1 + pnl_ratio) * (1 - self.commission)
            risk = self.initial_sl - self.entry_price
            reward = self.entry_price - price
            self.active_margin -= margin_portion
            
        rr_ratio = reward / risk if risk > 0 else 0
        self.balance += proceeds
        self.shares_remaining -= shares_to_close
        self.tp_closed_levels[tp_level] = True
        self.tp_count += 1
        
        self.partial_closes.append({
            'tp_level': tp_level + 1, 'price': price, 'pnl_ratio': pnl_ratio, 'rr_ratio': rr_ratio
        })
        
        self._log_partial_close(price, pnl_ratio, tp_level, rr_ratio)

    def _log_partial_close(self, price: float, pnl_ratio: float, tp_level: int, rr_ratio: float):
        """Логирование частичного закрытия"""
        try:
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.current_step, "LONG_PARTIAL" if self.position == 1 else "SHORT_PARTIAL",
                    round(self.entry_price, 4), round(self.initial_sl, 4), round(self.current_sl, 4),
                    ",".join([f"{p:.4f}" for p in self.tp_prices]), round(price, 4),
                    f"{pnl_ratio*100:.2f}%", round(self.net_worth, 2), f"TP_LEVEL_{tp_level+1}",
                    self.steps_since_open, "YES" if self.trailing_active else "NO",
                    str(tp_level+1), len(self.partial_closes), "GOOD" if pnl_ratio > 0.01 else "NORMAL",
                    f"{rr_ratio:.2f}"
                ])
        except: pass

    def _update_trailing_stop(self, current_price: float, atr: float):
        """Обновление трейлинг-стопа"""
        if self.position == 0: return
        
        if self.position == 1:
            profit_pct = (current_price - self.entry_price) / self.entry_price
            self.highest_profit_pct = max(self.highest_profit_pct, profit_pct)
            if profit_pct >= (atr / self.entry_price) * self.trailing_activation_atr:
                self.trailing_active = True
                mult = self.trailing_distance_atr
                if profit_pct > 0.02: mult *= 0.85
                elif profit_pct > 0.015: mult *= 0.90
                self.current_sl = max(self.current_sl, current_price - (atr * mult))
            if self.trailing_active and self.highest_profit_pct > 0:
                if (self.highest_profit_pct - profit_pct) / self.highest_profit_pct > 0.4:
                    self.current_sl = max(self.current_sl, current_price - (atr * self.protective_trailing_atr))
        else:
            profit_pct = (self.entry_price - current_price) / self.entry_price
            self.lowest_profit_pct = min(self.lowest_profit_pct, -profit_pct)
            if profit_pct >= (atr / self.entry_price) * self.trailing_activation_atr:
                self.trailing_active = True
                mult = self.short_config.get('trailing_distance_atr', self.trailing_distance_atr)
                if profit_pct > 0.02: mult *= 0.85
                elif profit_pct > 0.015: mult *= 0.90
                self.current_sl = min(self.current_sl, current_price + (atr * mult))
            if self.trailing_active and abs(self.lowest_profit_pct) > 0:
                if (abs(self.lowest_profit_pct) - abs(profit_pct)) / abs(self.lowest_profit_pct) > 0.4:
                    self.current_sl = min(self.current_sl, current_price + (atr * self.protective_trailing_atr))

    def _check_full_close(self, current_price: float) -> bool:
        """Проверка полного закрытия"""
        if self.position == 1:
            if current_price <= self.current_sl:
                self.exit_type = "SL_TRAILING" if self.trailing_active else "SL_INITIAL"
                self.actual_exit_price = min(self.current_sl, current_price * 0.998)
                self.sl_count += 1
                return True
        elif self.position == -1:
            if current_price >= self.current_sl:
                self.exit_type = "SL_TRAILING" if self.trailing_active else "SL_INITIAL"
                self.actual_exit_price = max(self.current_sl, current_price * 1.002)
                self.sl_count += 1
                return True
        if all(self.tp_closed_levels):
            self.exit_type = "TP_FULL"
            self.actual_exit_price = current_price
            self.tp_count += 1
            return True
        return False

    def _close_position(self, exec_price: float):
        """Закрытие позиции"""
        trade_type = "LONG" if self.position == 1 else "SHORT"
        if self.position == 1:
            final_price = exec_price * (1 - self.slippage)
            partial_pnl = sum(pc['pnl_ratio'] * (pc.get('shares_closed', 0) / self.shares_held) for pc in self.partial_closes) if self.shares_held > 0 else 0
            remaining_pnl = ((final_price - self.entry_price) / self.entry_price) * (self.shares_remaining / self.shares_held) if self.shares_held > 0 else 0
            total_pnl = partial_pnl + remaining_pnl
            if self.shares_remaining > 0:
                self.balance += self.shares_remaining * final_price * (1 - self.commission)
            risk = self.entry_price - self.initial_sl
            reward = (self.tp_prices[0] if self.tp_prices else final_price) - self.entry_price
        else:
            final_price = exec_price * (1 + self.slippage)
            total_pnl = (self.entry_price - final_price) / self.entry_price
            if self.active_margin > 0:
                self.balance += self.active_margin * (1 + total_pnl) * (1 - self.commission)
            risk = self.initial_sl - self.entry_price
            reward = self.entry_price - (self.tp_prices[0] if self.tp_prices else final_price)
            
        rr_ratio = reward / risk if risk > 0 else 0
        self.net_worth = self.balance
        
        quality = "NORMAL"
        if total_pnl > 0.02: quality = "EXCELLENT"
        elif total_pnl > 0.008: quality = "GOOD"
        elif total_pnl > -0.005: quality = "BAD"
        elif total_pnl <= -0.005: quality = "VERY_BAD"
        
        self._log_trade_final(final_price, total_pnl * 100, trade_type, quality, rr_ratio)
        
        self.total_trades += 1
        self.total_pnl += total_pnl
        self.recent_trades_pnl.append(total_pnl)
        if len(self.recent_trades_pnl) > self.max_recent_trades: self.recent_trades_pnl.pop(0)
        
        if total_pnl > 0:
            self.winning_trades += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.win_streak += 1
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.win_streak = 0
            
        self.trade_history.append({
            'step': self.current_step, 'type': trade_type, 'pnl': total_pnl, 'quality': quality, 'rr_ratio': rr_ratio
        })
        
        self.position = 0
        self.shares_held = 0
        self.shares_remaining = 0
        self.active_margin = 0
        self.tp_prices = []
        self.partial_closes = []
        self.tp_closed_levels = [False, False, False]
        self.trailing_active = False

    def _log_trade_final(self, exit_price: float, pnl_pct: float, trade_type: str, quality: str, rr_ratio: float):
        """Финальное логирование сделки"""
        reason = self.exit_type or "UNKNOWN"
        tp_info = ",".join([str(i+1) for i, c in enumerate(self.tp_closed_levels) if c]) or "NONE"
        try:
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.current_step, trade_type, round(self.entry_price, 4), round(self.initial_sl, 4),
                    round(self.current_sl, 4), ",".join([f"{p:.4f}" for p in self.tp_prices]),
                    round(exit_price, 4), f"{pnl_pct:.2f}%", round(self.net_worth, 2), reason,
                    self.steps_since_open, "YES" if self.trailing_active else "NO", tp_info,
                    len(self.partial_closes), quality, f"{rr_ratio:.2f}"
                ])
        except: pass

    def _log_open_position(self, side: str, price: float, atr: float):
        """Логирование открытия"""
        if not self.log_open_positions: return
        try:
            entry = self.entry_price
            sl = self.initial_sl
            tp1 = self.tp_prices[0] if self.tp_prices else 0
            risk = abs(entry - sl)
            reward = abs(tp1 - entry)
            rr_tp1 = reward / risk if risk > 0 else 0
            
            if self.open_log_file:
                with open(self.open_log_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        self.current_step, side, round(entry, 6), round(sl, 6),
                        round(tp1, 6), round(self.tp_prices[1], 6) if len(self.tp_prices)>1 else 0,
                        round(self.tp_prices[2], 6) if len(self.tp_prices)>2 else 0,
                        f"{rr_tp1:.4f}", f"{atr:.6f}", f"{atr/price:.6f}",
                        f"{self.margin_percent:.6f}", f"{self.active_margin:.2f}", f"{self.shares_held:.6f}"
                    ])
        except: pass

    def _should_close_by_action(self, action: int, prev_position: int) -> bool:
        """Закрытие по сигналу модели"""
        if self.steps_since_open < self.min_hold_steps: return False
        if (action == 1 and prev_position == -1) or (action == 2 and prev_position == 1):
            return self.steps_since_open >= self.min_hold_steps * 2
        return False

    def _update_net_worth(self, current_price: float):
        """Обновление Net Worth"""
        if self.position == 1:
            self.net_worth = self.balance + (self.shares_remaining * current_price)
        elif self.position == -1:
            pnl_ratio = (self.entry_price - current_price) / self.entry_price
            self.net_worth = self.balance + (self.active_margin * (1 + pnl_ratio))
        else:
            self.net_worth = self.balance

    def _update_quality_stats(self, reward: float, trade_closed: bool, partial_close: bool):
        """Обновление статистики качества"""
        if trade_closed and len(self.recent_trades_pnl) > 0:
            self.avg_profit_last_10 = np.mean(self.recent_trades_pnl[-10:])
            if self.recent_trades_pnl[-1] > 0:
                self.consecutive_profitable_trades += 1
                self.consecutive_loss_trades = 0
            else:
                self.consecutive_loss_trades += 1
                self.consecutive_profitable_trades = 0

    def _get_info(self) -> Dict:
        """Информация о состоянии"""
        trades = self.total_trades or 1
        win_rate = self.winning_trades / trades
        drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth if self.max_net_worth > 0 else 0
        return {
            'net_worth': self.net_worth, 'balance': self.balance, 'position': self.position,
            'total_trades': self.total_trades, 'win_rate': win_rate, 'total_pnl': self.total_pnl,
            'drawdown': drawdown, 'trades_today': self.trades_today, 'current_day': self.current_day
        }

    def render(self, mode='human'):
        """Отрисовка"""
        if mode == 'human':
            print(f"Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, Pos: {self.position}")

    # --- MTF Специфичные методы ---

    def _validate_mtf_sync(self):
        """Проверка синхронизации временных меток"""
        if self.df_1h is None: return
        for d in [self.df, self.df_1h]:
            if 'timestamp' not in d.columns and isinstance(d.index, pd.DatetimeIndex):
                d['timestamp'] = d.index
        if 'timestamp' in self.df.columns and 'timestamp' in self.df_1h.columns:
            t1 = pd.to_datetime(self.df['timestamp'].iloc[0])
            t2 = pd.to_datetime(self.df_1h['timestamp'].iloc[0])
            if abs((t1 - t2).total_seconds()) > 3600 * 24:
                print(f"⚠️ [MTF] ВНИМАНИЕ: Большая разница в начале данных ТФ!")

    def _find_nearest_idx(self, df_target: pd.DataFrame, timestamp: pd.Timestamp) -> int:
        """Поиск ближайшего индекса в другом ТФ"""
        if df_target is None or len(df_target) == 0: return 0
        cache_key = (id(df_target), timestamp)
        if cache_key in self._mtf_index_cache: return self._mtf_index_cache[cache_key]
        
        if 'timestamp' in df_target.columns:
            times = pd.to_datetime(df_target['timestamp'])
        else:
            times = df_target.index
            
        mask = times <= timestamp
        result = int(np.argmax(mask.values)) if mask.any() else 0
        if len(self._mtf_index_cache) < 1000: self._mtf_index_cache[cache_key] = result
        return result

    def _count_mtf_features(self) -> int:
        """Количество MTF признаков"""
        return 19

    def _get_mtf_features(self, current_idx: int) -> np.ndarray:
        """Извлечение MTF признаков"""
        if not self.mtf_enabled: return np.zeros(19, dtype=np.float32)
        try:
            ts = self.df.iloc[current_idx]['timestamp']
            current_time = pd.to_datetime(ts)
            features = []
            
            # 1h features (6)
            idx_1h = self._find_nearest_idx(self.df_1h, current_time)
            row_1h = self.df_1h.iloc[idx_1h]
            features.extend([
                float(row_1h.get('adx', 25))/100, float(row_1h.get('plus_di', 25))/100,
                float(row_1h.get('minus_di', 25))/100, float(row_1h.get('rsi', 50))/100,
                (float(row_1h.get('atr', 0))/float(row_1h.get('close', 1))),
                float(row_1h.get('volume', 0))/1000000
            ])
            
            # 4h features (5)
            idx_4h = self._find_nearest_idx(self.df_4h, current_time)
            row_4h = self.df_4h.iloc[idx_4h]
            features.extend([
                float(row_4h.get('adx', 25))/100, float(row_4h.get('plus_di', 25))/100,
                float(row_4h.get('minus_di', 25))/100, float(row_4h.get('rsi', 50))/100,
                (float(row_4h.get('close', 0))/float(row_1h.get('close', 1)))
            ])
            
            # Alignment (3), Conflict (1), Zones (4)
            features.extend(self._calculate_trend_alignment(current_time))
            features.append(self._calculate_conflict_score(current_time))
            features.extend(self._analyze_zones(current_time))
            
            return np.array(features, dtype=np.float32)
        except: return np.zeros(19, dtype=np.float32)

    def _calculate_trend_alignment(self, timestamp: pd.Timestamp) -> List[float]:
        """Выравнивание трендов"""
        try:
            idx_1h = self._find_nearest_idx(self.df_1h, timestamp)
            idx_4h = self._find_nearest_idx(self.df_4h, timestamp)
            row_15m = self.df.iloc[self.current_step]
            row_1h = self.df_1h.iloc[idx_1h]
            row_4h = self.df_4h.iloc[idx_4h]
            
            t15 = 1.0 if row_15m.get('plus_di', 25) > row_15m.get('minus_di', 25) else -1.0
            t1h = 1.0 if row_1h.get('plus_di', 25) > row_1h.get('minus_di', 25) else -1.0
            t4h = 1.0 if row_4h.get('plus_di', 25) > row_4h.get('minus_di', 25) else -1.0
            
            return [t15 * t1h, t1h * t4h, (t15 * t1h * t4h) / 3.0]
        except: return [0.0, 0.0, 0.0]

    def _calculate_conflict_score(self, timestamp: pd.Timestamp) -> float:
        """Оценка конфликта"""
        align = self._calculate_trend_alignment(timestamp)
        return -align[2]

    def _analyze_zones(self, timestamp: pd.Timestamp) -> List[float]:
        """Анализ зон RSI"""
        try:
            idx_1h = self._find_nearest_idx(self.df_1h, timestamp)
            idx_4h = self._find_nearest_idx(self.df_4h, timestamp)
            z15 = (float(self.df.iloc[self.current_step].get('rsi', 50)) - 50) / 50
            z1h = (float(self.df_1h.iloc[idx_1h].get('rsi', 50)) - 50) / 50
            z4h = (float(self.df_4h.iloc[idx_4h].get('rsi', 50)) - 50) / 50
            return [z15, z1h, z4h, (z15 + z1h + z4h) / 3]
        except: return [0.0, 0.0, 0.0, 0.0]

    def _get_observation(self) -> np.ndarray:
        """Получение наблюдения"""
        if len(self.df) == 0: return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        market_data = []
        for col in self.obs_cols:
            val = float(self.df.loc[self.current_step, col]) if col in self.df.columns else 0.0
            market_data.append(val * self.obs_scaling.get(col, 1.0))
            
        trade_quality = 0.0
        if self.recent_trades_pnl:
            trade_quality = np.mean(self.recent_trades_pnl[-5:]) * 100 + (sum(1 for p in self.recent_trades_pnl[-5:] if p > 0) / min(5, len(self.recent_trades_pnl)))
            
        position_state = [
            self.position, min(1.0, self.steps_since_open / 100.0), min(1.0, self.steps_since_last_trade / 50.0),
            self.consecutive_losses / 5.0, self.consecutive_wins / 5.0, (self.net_worth - self.initial_balance) / self.initial_balance,
            min(1.0, max(0.0, (self.max_net_worth - self.net_worth) / max(self.max_net_worth, 1e-9))),
            min(1.0, self.active_margin / max(self.balance, 1e-9)), self.consecutive_profitable_trades / 10.0,
            self.consecutive_loss_trades / 10.0, trade_quality / 2.0, min(2.0, max(-1.0, self.avg_profit_last_10))
        ]
        
        base_obs = np.concatenate([market_data, position_state])
        mtf_obs = self._get_mtf_features(self.current_step)
        obs = np.concatenate([base_obs, mtf_obs]).astype(np.float32)
        return np.nan_to_num(obs)

    def _check_entry_filters_strict(self, price: float, atr: float, action: int = None) -> bool:
        """Ужесточенные фильтры входа"""
        if self.current_step >= len(self.df): return False
        try:
            atr_pct = atr / price
            if atr_pct < self.mtf_atr_percent_min or atr_pct > 0.04 or atr < self.mtf_min_absolute_atr: return False
            
            row = self.df.loc[self.current_step]
            adx = float(row.get('adx', 0))
            min_adx = self.min_adx if action == 1 else self.mtf_min_adx_short
            if adx < min_adx: return False
            
            if action == 1 and float(row.get('plus_di', 0)) <= float(row.get('minus_di', 0)): return False
            if action == 2 and float(row.get('minus_di', 0)) <= float(row.get('plus_di', 0)) * 0.95: return False
            
            rsi_norm = abs(float(row.get('rsi_norm', 0)))
            cfg = self.long_config if action == 1 else self.short_config
            if rsi_norm < cfg['min_rsi_norm'] or rsi_norm > cfg['max_rsi_norm']: return False
            
            vol = float(row.get('volume', 0))
            if vol < self.mtf_min_absolute_volume: return False
            if self.current_step >= 20:
                avg_vol = self.df.loc[self.current_step-20:self.current_step, 'volume'].mean()
                if vol / avg_vol < (self.mtf_min_volume_spike if action == 1 else self.mtf_min_volume_spike_short): return False
                
            v_ratio = float(row.get('volatility_ratio', 1.0))
            if v_ratio < self.min_volatility_ratio or v_ratio > self.max_volatility_ratio: return False
            
            sl_dist = np.clip(atr * self.atr_multiplier, price * self.min_sl_percent, price * self.max_sl_percent)
            min_tp = max(sl_dist * self.min_rr_ratio, atr * self.tp_levels[0], price * self.min_tp_percent)
            if min_tp / sl_dist < self.min_rr_ratio: return False
            if min_tp / price > (0.02 if action == 1 else 0.03): return False
            
            if self.mtf_enabled and not self._check_mtf_entry_filters(action): return False
            
            self.rr_stats.append(min_tp / sl_dist)
            if len(self.rr_stats) > 100: self.rr_stats.pop(0)
            return True
        except: return False

    def _check_mtf_entry_filters(self, action: int) -> bool:
        """MTF фильтры входа"""
        try:
            ts = pd.to_datetime(self.df.iloc[self.current_step]['timestamp'])
            side = 'LONG' if action == 1 else 'SHORT'
            if not self._check_1h_trend(ts, side): return False
            if not self._check_4h_trend(ts, side): return False
            if self._check_trend_conflict(ts, side): return False
            return True
        except: return True

    def _check_1h_trend(self, ts: pd.Timestamp, side: str) -> bool:
        """Тренд 1h"""
        if self.df_1h is None: return True
        idx = self._find_nearest_idx(self.df_1h, ts)
        row = self.df_1h.iloc[idx]
        adx, pdi, mdi = float(row.get('adx', 0)), float(row.get('plus_di', 25)), float(row.get('minus_di', 25))
        if side == 'LONG': return adx >= self.mtf_min_1h_adx and pdi > mdi * self.mtf_di_ratio_1h
        return adx >= self.mtf_min_1h_adx_short and mdi > pdi * self.mtf_di_ratio_1h

    def _check_4h_trend(self, ts: pd.Timestamp, side: str) -> bool:
        """Тренд 4h"""
        if self.df_4h is None: return True
        idx = self._find_nearest_idx(self.df_4h, ts)
        row = self.df_4h.iloc[idx]
        pdi, mdi = float(row.get('plus_di', 25)), float(row.get('minus_di', 25))
        return pdi > mdi if side == 'LONG' else mdi > pdi

    def _check_trend_conflict(self, ts: pd.Timestamp, side: str) -> bool:
        """Конфликт трендов"""
        if not self.mtf_enabled: return False
        idx1 = self._find_nearest_idx(self.df_1h, ts)
        idx4 = self._find_nearest_idx(self.df_4h, ts)
        r1, r4 = self.df_1h.iloc[idx1], self.df_4h.iloc[idx4]
        p1, m1, a1 = float(r1.get('plus_di', 25)), float(r1.get('minus_di', 25)), float(r1.get('adx', 0))
        p4, m4, a4 = float(r4.get('plus_di', 25)), float(r4.get('minus_di', 25)), float(r4.get('adx', 0))
        
        if side == 'LONG':
            against1, against4 = m1 > p1 * 1.05, m4 > p4 * 1.05
            neut1, neut4 = abs(p1-m1) < 2 or a1 < 20, abs(p4-m4) < 2 or a4 < 20
        else:
            against1, against4 = p1 > m1 * 1.05, p4 > m4 * 1.05
            neut1, neut4 = abs(p1-m1) < 2 or a1 < 20, abs(p4-m4) < 2 or a4 < 20
            
        return (against1 and against4) or (against1 and neut4) or (against4 and neut1) or (neut1 and neut4)

    def _open_long_with_tp_features(self, price: float, atr: float):
        """Открытие LONG"""
        self.entry_price = price * (1 + self.slippage)
        sl_dist = np.clip(atr * self.atr_multiplier, price * self.min_sl_percent, price * self.max_sl_percent)
        self.initial_sl = self.entry_price - sl_dist
        self.current_sl = self.initial_sl
        base_tp = max(sl_dist * self.min_rr_ratio, atr * self.tp_levels[0], price * self.min_tp_percent)
        self.tp_prices = [self.entry_price + base_tp * (m / self.tp_levels[0]) for m in self.tp_levels]
        self._setup_trade_enhanced(1)
        self._log_open_position("LONG", price, atr)

    def _open_short_with_tp_features(self, price: float, atr: float):
        """Открытие SHORT"""
        self.entry_price = price * (1 - self.slippage)
        sl_dist = np.clip(atr * self.atr_multiplier, price * self.min_sl_percent, price * self.max_sl_percent)
        self.initial_sl = self.entry_price + sl_dist
        self.current_sl = self.initial_sl
        base_tp = max(sl_dist * self.min_rr_ratio, atr * self.tp_levels[0], price * self.min_tp_percent)
        self.tp_prices = [self.entry_price - base_tp * (m / self.tp_levels[0]) for m in self.tp_levels]
        self._setup_trade_enhanced(-1)
        self._log_open_position("SHORT", price, atr)

    def _calculate_reward_profit_focused(self, prev_net_worth: float, trade_opened: bool, trade_closed: bool, partial_close: bool, current_price: float, action: int) -> float:
        """Reward функция"""
        reward = np.tanh(((self.net_worth - prev_net_worth) / self.initial_balance) * 40.0) * 1.5
        
        if trade_opened:
            row = self.df.iloc[self.current_step]
            adx, pdi, mdi = row.get('adx', 0), row.get('plus_di', 0), row.get('minus_di', 0)
            if self.position == 1:
                reward += 4.0 if (adx >= self.min_adx and pdi > mdi) else -3.0 if (adx >= self.min_adx and mdi > pdi) else 0
            else:
                reward += 6.0 if (adx >= self.min_adx and mdi > pdi) else 3.0 if (adx >= 20 and mdi > pdi) else -3.0 if (adx >= self.min_adx and pdi > mdi) else 0
            
            if self.mtf_enabled:
                ts = pd.to_datetime(row['timestamp'])
                if self._entered_at_trend_start(ts): reward += 3.0
                side = 'LONG' if action == 1 else 'SHORT'
                if not self._check_trend_conflict(ts, side): reward += 2.0
                reward += self._get_zone_alignment(ts, side) * 1.5
                if self._ignored_strong_trend(ts, side): reward -= 4.0

        if action == 0 and self.position == 0:
            try:
                if self._check_entry_filters_strict(current_price, float(self.df.iloc[self.current_step]['atr'])):
                    reward -= 1.0
            except: pass

        if partial_close and self.partial_closes:
            last = self.partial_closes[-1]
            reward += 18.0 * (last['tp_level'] * 0.9) + min(30.0, last['pnl_ratio'] * 400)

        elif trade_closed and self.exit_type in ["SL_INITIAL", "SL_TRAILING"]:
            pnl_sl = (self.net_worth / prev_net_worth) - 1
            if self.exit_type == "SL_TRAILING":
                self.recent_trailing_sl.append(1)
                if len(self.recent_trailing_sl) > 10: self.recent_trailing_sl.pop(0)
                if len(self.recent_trailing_sl) >= 5 and sum(self.recent_trailing_sl)/len(self.recent_trailing_sl) > 0.5:
                    reward -= 0.2
            
            if self.trade_history:
                rr = self.trade_history[-1].get('rr_ratio', 1.0)
                reward += rr * 0.1 if rr > 2.0 else -3.0 * (1.0 - rr) if rr < 1.0 else 0
            
            reward -= 12.0 if pnl_sl < -0.02 else 8.0 if pnl_sl < -0.01 else 6.0

        if trade_closed and self.trade_history and self.trade_history[-1].get('quality') == 'VERY_BAD':
            reward -= 5.0
            
        if len(self.rr_stats) >= 5:
            avg_rr = np.mean(self.rr_stats[-5:])
            reward += min(10.0, (avg_rr - 2.0) * 3.0) if avg_rr > 2.0 else -(1.5 - avg_rr) * 2.0 if avg_rr < 1.5 else 0
            
        if trade_closed and self.recent_trades_pnl:
            if self.recent_trades_pnl[-1] > 0: reward += self.win_streak * 0.05
            if len(self.recent_trades_pnl) >= 3 and all(p > 0 for p in self.recent_trades_pnl[-3:]): reward += 5.0
            
        return np.clip(reward, -15.0, 35.0)

    def _entered_at_trend_start(self, ts: pd.Timestamp) -> bool:
        """Вход в начале тренда 4h"""
        if self.df_4h is None: return False
        idx = self._find_nearest_idx(self.df_4h, ts)
        if idx < 5: return False
        return float(self.df_4h.iloc[idx].get('adx', 0)) > float(self.df_4h.iloc[idx-5].get('adx', 0)) * 1.1

    def _get_zone_alignment(self, ts: pd.Timestamp, side: str) -> float:
        """Выравнивание зон"""
        try:
            zones = self._analyze_zones(ts)
            consensus = zones[3]
            return max(0.0, -consensus) if side == 'LONG' else max(0.0, consensus)
        except: return 0.5

    def _ignored_strong_trend(self, ts: pd.Timestamp, side: str) -> bool:
        """Игнорирование сильного тренда 4h"""
        if self.df_4h is None: return False
        idx = self._find_nearest_idx(self.df_4h, ts)
        row = self.df_4h.iloc[idx]
        if float(row.get('adx', 0)) < 30: return False
        p, m = float(row.get('plus_di', 25)), float(row.get('minus_di', 25))
        return p > m * 1.2 if side == 'LONG' else m > p * 1.2
