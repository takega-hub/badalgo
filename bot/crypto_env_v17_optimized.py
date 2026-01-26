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
        
        # TP уровни: снижены для большего количества достижений TP (ПО РЕКОМЕНДАЦИЯМ АНАЛИЗА)
        # Анализ показал: только 9.3% сделок закрываются по TP_LEVEL_1, нужно больше TP закрытий
        self.tp_levels = [1.8, 2.5, 3.5]  # РЕКОМЕНДАЦИЯ: снизить с [2.0, 3.0, 4.0] для большего % TP закрытий
        self.tp_close_percentages = [0.25, 0.35, 0.40]  # Больше на последних уровнях
        
        # Трейлинг-стоп: настроен для уменьшения ложных срабатываний (ПО РЕКОМЕНДАЦИЯМ АНАЛИЗА)
        # Анализ показал: 41.1% сделок закрываются по SL_TRAILING - требуется дальнейшая оптимизация
        self.trailing_activation_atr = 0.30   # РЕКОМЕНДАЦИЯ: увеличено с 0.20 до 0.25-0.30 - позже активация для уменьшения SL_TRAILING
        self.trailing_distance_atr = 0.40     # РЕКОМЕНДАЦИЯ: увеличено с 0.30 до 0.35-0.40 - больше расстояние для уменьшения ложных срабатываний
        self.protective_trailing_atr = 0.5    # Защитный стоп (было 0.6)
        # Время удержания
        self.max_hold_steps = 60
        self.min_hold_steps = 8
        
        # УЖЕСТОЧЕННЫЕ ФИЛЬТРЫ ВХОДА (УЛУЧШЕНО)
        self.min_sl_percent = 0.003           # Минимальный SL 0.3% (оставляем как есть)
        self.max_sl_percent = 0.007           # УМЕНЬШЕНО с 0.008 до 0.007 - более строгий SL

        self.min_tp_percent = 0.006          # УМЕНЬШЕНО с 0.008 до 0.006 - TP уровни уже снижены до [1.8, 2.5, 3.5]
        
        # Маржинальность: консервативная
        self.base_margin_percent = 0.07
        
        # Лимит сделок: меньше сделок, выше качество
        self.max_daily_trades = 5
        self.trades_today = 0
        self.current_day = 0
        
        # УЖЕСТОЧЕННЫЕ ФИЛЬТРЫ ДЛЯ КАЧЕСТВЕННОГО ВХОДА (ПО РЕКОМЕНДАЦИЯМ АНАЛИЗА)
        # Анализ показал: Win Rate 48.4% (цель ≥50%), LONG WR 34.3%, SHORT WR 27.4%
        # Проблемы: много SL_TRAILING (37.9%), много VERY_BAD сделок (28.2%)
        # Используем ADX (Average Directional Index) - стандартный индикатор силы тренда
        # ADX > 25 = сильный тренд, ADX > 30 = очень сильный тренд
        self.min_adx = 25.0                  # Минимальный ADX для входа (сильный тренд)
        self.min_trend_strength = 0.55        # УЖЕСТОЧЕНО: увеличено до 0.55 (рекомендация: 0.50-0.55)
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
        
        # LONG_CONFIG: оптимизированные параметры для LONG позиций
        self.long_config = {
            'min_trend_strength': 0.50,          # стандартный фильтр
            'min_rsi_norm': 0.15,                # перепроданность (RSI ~15-40)
            'max_rsi_norm': 0.60,                # не выше RSI 60
            'trailing_distance_atr': 0.35,       # стандартное расстояние
            'position_size_multiplier': 1.0,     # полный размер позиции
        }
        
        # SHORT_CONFIG: более строгие параметры для SHORT позиций (работает хуже)
        self.short_config = {
            'min_trend_strength': 0.60,          # более строгий фильтр
            'min_rsi_norm': 0.55,                # только перекупленность (RSI ~55-85)
            'max_rsi_norm': 0.85,                # максимум RSI 85
            'trailing_distance_atr': 0.40,       # больше расстояние (больший стоп-лосс)
            'position_size_multiplier': 0.7,     # меньший размер позиции (меньший риск)
        }
        
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
        
        # SHORT_CONFIG: более строгие параметры для SHORT позиций (работает хуже)
        self.short_config = {
            'min_trend_strength': 0.60,          # более строгий фильтр
            'min_rsi_norm': 0.55,                # только перекупленность (RSI ~55-85)
            'max_rsi_norm': 0.85,                # максимум RSI 85
            'trailing_distance_atr': 0.40,       # больше расстояние (больший стоп-лосс)
            'position_size_multiplier': 0.7,     # меньший размер позиции (меньший риск)
        }
        
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
                self._close_position(self.actual_exit_price)
                trade_closed = True
            
            elif self._should_close_by_action(action, prev_position):
                if self.steps_since_open >= self.max_hold_steps * 0.8:
                    self.exit_type = "MANUAL"
                    self._close_position(current_price)
                    trade_closed = True
                    self.manual_count += 1
                    print(f"[MANUAL] Закрытие по действию")
        
        # 2. Открытие новой позиции - ТОЛЬКО ЕСЛИ ВСЕ ФИЛЬТРЫ ПРОЙДЕНЫ
        if not trade_closed and self.position == 0:
            if self.steps_since_last_trade >= 10:
                if self.trades_today < self.max_daily_trades:
                    
                    # ЖЕСТКИЙ ФИЛЬТР ВХОДА С ГАРАНТИЕЙ RR
                    can_enter = self._check_entry_filters_strict(current_price, current_atr, action=action)
                    
                    if can_enter:
                        prev_pos_before_open = self.position
                        # Логи отключены для уменьшения спама
                        
                        if action == 1:  # Long
                            self._open_long_with_tp_features(current_price, current_atr)
                            if self.position != 1:
                                print(f"⚠️ [ERROR] Position mismatch! Expected 1 (LONG), got {self.position}")
                            trade_opened = True
                            self.trades_today += 1
                        elif action == 2:  # Short
                            self._open_short_with_tp_features(current_price, current_atr)
                            if self.position != -1:
                                print(f"⚠️ [ERROR] Position mismatch! Expected -1 (SHORT), got {self.position}")
                            trade_opened = True
                            self.trades_today += 1
                        # Логи отключены

        # 3. Обновление временных метрик
        if self.position == 0 and not trade_closed:
            self.steps_since_last_trade += 1
        
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
        """УЖЕСТОЧЕННЫЕ фильтры для входа с гарантией RR ≥ 1.5"""
        if self.current_step >= len(self.df):
            return False
        
        try:
            # 1. Фильтр по волатильности (ATR)
            atr_percent = atr / price
            if atr_percent < 0.001 or atr_percent > 0.04:  # Более гибкий диапазон
                return False
            
            # 2. Проверка силы тренда через ADX
            # trend_bias_1h УБРАН ИЗ ФИЛЬТРОВ (отрицательная корреляция: -0.0345)
            # Используем ADX (Average Directional Index) - стандартный индикатор силы тренда
            # ADX > 25 = сильный тренд, ADX > 30 = очень сильный тренд
            if 'adx' in self.df.columns:
                try:
                    adx_value = float(self.df.loc[self.current_step, 'adx'])
                    if adx_value < self.min_adx:
                        return False  # Слишком слабый тренд
                    
                    # Дополнительно: проверка направления через +DI и -DI (если доступны)
                    if action is not None:
                        if 'plus_di' in self.df.columns and 'minus_di' in self.df.columns:
                            try:
                                plus_di = float(self.df.loc[self.current_step, 'plus_di'])
                                minus_di = float(self.df.loc[self.current_step, 'minus_di'])
                                
                                if action == 1:  # LONG - +DI должен быть больше -DI
                                    if plus_di <= minus_di:
                                        return False  # Нисходящий тренд, не открываем LONG
                                elif action == 2:  # SHORT - -DI должен быть больше +DI
                                    if minus_di <= plus_di:
                                        return False  # Восходящий тренд, не открываем SHORT
                            except:
                                pass  # Если DI недоступны, пропускаем проверку направления
                except:
                    return False
            else:
                # Fallback: проверка ATR если ADX недоступен
                atr_percent = atr / price
                if atr_percent < 0.0015:  # Минимальный ATR для входа
                    return False
            
            # 3. РАЗДЕЛЬНАЯ ПРОВЕРКА RSI ДЛЯ LONG/SHORT (ОПТИМИЗИРОВАНО)
            # КРИТИЧНО: rsi_norm имеет сильную корреляцию 0.2229 с PnL и влияет на WR (разница 36.8%!)
            # Для LONG: перепроданность (низкий RSI) - вход в зоне 0.15-0.60
            # Для SHORT: перекупленность (высокий RSI) - вход в зоне 0.55-0.85
            if 'rsi_norm' in self.df.columns:
                try:
                    rsi_norm = float(self.df.loc[self.current_step, 'rsi_norm'])
                    rsi_norm_abs = abs(rsi_norm)
                    
                    # Раздельные фильтры для LONG и SHORT
                    if action == 1:  # LONG позиция
                        config = self.long_config
                        if rsi_norm_abs < config['min_rsi_norm'] or rsi_norm_abs > config['max_rsi_norm']:
                            return False  # LONG: только в зоне перепроданности (0.15-0.60)
                    elif action == 2:  # SHORT позиция
                        config = self.short_config
                        if rsi_norm_abs < config['min_rsi_norm'] or rsi_norm_abs > config['max_rsi_norm']:
                            return False  # SHORT: только в зоне перекупленности (0.55-0.85)
                    else:
                        # Для других действий используем общий фильтр
                        if rsi_norm_abs < 0.15 or rsi_norm_abs > 0.85:
                            return False
                except:
                    pass
            
            # 4. Проверка объема с всплеском (КРИТИЧНО: анализ показал корреляцию 0.1581 и разницу 31.5%!)
            # Требуем всплеск объёма >= 1.5x среднего для лучшего входа
            if 'volume' in self.df.columns:
                try:
                    current_volume = float(self.df.loc[self.current_step, 'volume'])
                    # Вычисляем средний объем за последние 20 свечей
                    if self.current_step >= 20:
                        avg_volume = float(self.df.loc[self.current_step-20:self.current_step, 'volume'].mean())
                        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                        # Требуем всплеск объёма (анализ показал важность объёма)
                        if volume_ratio < self.min_volume_spike:
                            return False  # Недостаточный всплеск объёма
                except:
                    return False
            
            # 5. Проверка движения цены от экстремума (опционально)
            # Из анализа: close имеет разницу 2.2% между прибыльными/убыточными
            if self.current_step >= 10:
                try:
                    current_price = float(self.df.loc[self.current_step, 'close'])
                    # Находим экстремум за последние 10 свечей
                    recent_high = float(self.df.loc[self.current_step-10:self.current_step, 'high'].max())
                    recent_low = float(self.df.loc[self.current_step-10:self.current_step, 'low'].min())
                    
                    if action == 1:  # LONG - проверяем расстояние от минимума
                        distance_from_low = ((current_price - recent_low) / recent_low) * 100
                        if distance_from_low < self.min_price_distance_pct:
                            return False  # Слишком близко к минимуму
                    elif action == 2:  # SHORT - проверяем расстояние от максимума
                        distance_from_high = ((recent_high - current_price) / recent_high) * 100
                        if distance_from_high < self.min_price_distance_pct:
                            return False  # Слишком близко к максимуму
                except:
                    pass  # Если не удалось проверить, пропускаем
            
            # 5. КРИТИЧНО: Проверка volatility_ratio (анализ показал разницу Win Rate 15.9%!)
            if 'volatility_ratio' in self.df.columns:
                try:
                    volatility_ratio = float(self.df.loc[self.current_step, 'volatility_ratio'])
                    # Анализ показал: Q1 (низкая волатильность) = WR 15.8%, Q4 (высокая) = WR 61.4%
                    # Прибыльные сделки: volatility_ratio = 0.0046, убыточные = 0.0041
                    # Поэтому требуем volatility_ratio >= 0.0030 для входа (выше среднего убыточных сделок)
                    if volatility_ratio < self.min_volatility_ratio:
                        return False  # Слишком низкая волатильность = плохой Win Rate (Q1: 15.8% vs Q4: 61.4%)
                    if volatility_ratio > self.max_volatility_ratio:
                        return False  # Слишком высокая волатильность = риск
                except:
                    return False
            
            # 6. ГАРАНТИЯ MIN RR RATIO 1.5 - КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ!
            # Рассчитываем SL на основе ATR
            sl_distance = max(atr * self.atr_multiplier, price * self.min_sl_percent)
            sl_distance = min(sl_distance, price * self.max_sl_percent)
            
            # Минимальный TP для достижения RR 1.5
            min_tp_for_rr = sl_distance * self.min_rr_ratio
            
            # Берем максимальное из: минимальный TP для RR, ATR-based TP, процентный TP
            min_tp_distance = max(
                min_tp_for_rr,
                atr * self.tp_levels[0],
                price * self.min_tp_percent
            )
            
            # Проверяем фактический RR
            actual_rr = min_tp_distance / sl_distance if sl_distance > 0 else 0
            
            if actual_rr < self.min_rr_ratio:
                self.min_rr_violations += 1
                if self.min_rr_violations % 20 == 0:
                    print(f"[FILTER] RR violation {self.min_rr_violations}: {actual_rr:.2f} < {self.min_rr_ratio}")
                return False
            
            # 7. Дополнительная проверка: TP должен быть достижим
            tp_percent_needed = min_tp_distance / price
            if tp_percent_needed > 0.02:  # Если нужен TP > 2%, вероятно нереалистично
                return False
            
            # Сохраняем RR статистику
            self.rr_stats.append(actual_rr)
            if len(self.rr_stats) > 100:
                self.rr_stats.pop(0)
            
            return True
            
        except Exception as e:
            print(f"⚠️ Ошибка в фильтрах входа: {e}")
            return False
    
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
        
        try:
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.current_step,
                    "LONG_PARTIAL" if self.position == 1 else "SHORT_PARTIAL",
                    round(self.entry_price, 4),
                    round(self.initial_sl, 4),
                    round(self.current_sl, 4),
                    f"{self.tp_prices[0]:.4f},{self.tp_prices[1]:.4f},{self.tp_prices[2]:.4f}" if self.tp_prices else "",
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
        """Обновление трейлинг-стопа"""
        if self.position == 0:
            return
        
        if self.position == 1:
            profit_pct = (current_price - self.entry_price) / self.entry_price
            self.highest_profit_pct = max(self.highest_profit_pct, profit_pct)
            
            if profit_pct >= (atr / self.entry_price) * self.trailing_activation_atr:
                if not self.trailing_active:
                    self.trailing_active = True
                    # Логи отключены: print(f"[TRAILING_OPTIMIZED] Активация при прибыли {profit_pct*100:.2f}%")
                
                # Динамический трейлинг
                trailing_multiplier = self.trailing_distance_atr
                if profit_pct > 0.01:
                    trailing_multiplier *= 0.8
                
                trailing_stop_price = current_price - (atr * trailing_multiplier)
                self.current_sl = max(self.current_sl, trailing_stop_price)
            
            if self.trailing_active and self.highest_profit_pct > 0:
                current_drawdown = (self.highest_profit_pct - profit_pct) / self.highest_profit_pct
                if current_drawdown > 0.4:
                    protective_sl = current_price - (atr * self.protective_trailing_atr)
                    self.current_sl = max(self.current_sl, protective_sl)
        
        else:  # SHORT позиция
            profit_pct = (self.entry_price - current_price) / self.entry_price
            self.lowest_profit_pct = min(self.lowest_profit_pct, -profit_pct)
            
            # Используем отдельные параметры для SHORT (более строгие)
            trailing_activation = self.trailing_activation_atr
            trailing_distance = self.short_params.get('trailing_distance_atr', self.trailing_distance_atr)
            
            if profit_pct >= (atr / self.entry_price) * trailing_activation:
                if not self.trailing_active:
                    self.trailing_active = True
                    # Логи отключены: print(f"[TRAILING_OPTIMIZED] Активация при прибыли {profit_pct*100:.2f}%")
                
                trailing_multiplier = trailing_distance
                if profit_pct > 0.01:
                    trailing_multiplier *= 0.8
                
                trailing_stop_price = current_price + (atr * trailing_multiplier)
                self.current_sl = min(self.current_sl, trailing_stop_price)
            
            if self.trailing_active and abs(self.lowest_profit_pct) > 0:
                current_drawdown = (abs(self.lowest_profit_pct) - abs(profit_pct)) / abs(self.lowest_profit_pct)
                if current_drawdown > 0.4:
                    protective_sl = current_price + (atr * self.protective_trailing_atr)
                    self.current_sl = min(self.current_sl, protective_sl)
    
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
        
        try:
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.current_step,
                    trade_type,
                    round(self.entry_price, 4),
                    round(self.initial_sl, 4),
                    round(self.current_sl, 4),
                    f"{self.tp_prices[0]:.4f},{self.tp_prices[1]:.4f},{self.tp_prices[2]:.4f}" if self.tp_prices else "",
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
                rsi_norm = abs(float(row.get('rsi_norm', 0)))  # RSI для проверки правильности входа
                
                # БОНУС ЗА ПРАВИЛЬНЫЙ RSI ВХОД (НОВОЕ!)
                # Для LONG: перепроданность (RSI 0.15-0.60)
                # Для SHORT: перекупленность (RSI 0.55-0.85)
                if self.position == 1:  # LONG позиция
                    if self.long_config['min_rsi_norm'] <= rsi_norm <= self.long_config['max_rsi_norm']:
                        reward += 0.1  # Бонус за правильный RSI вход для LONG
                elif self.position == -1:  # SHORT позиция
                    if self.short_config['min_rsi_norm'] <= rsi_norm <= self.short_config['max_rsi_norm']:
                        reward += 0.1  # Бонус за правильный RSI вход для SHORT
                
                # БОНУС ЗА ОТКРЫТИЕ В ПРАВИЛЬНОМ НАПРАВЛЕНИИ ТРЕНДА (используем ADX и DI)
                if self.position == 1:  # LONG позиция
                    if adx_value >= self.min_adx and plus_di > minus_di:  # Сильный восходящий тренд
                        reward += 4.0
                        # Логи отключены: print(f"[REWARD] ✅ Бонус за LONG при сильном восходящем тренде: +4.0 (ADX={adx_value:.1f}, +DI={plus_di:.1f} > -DI={minus_di:.1f})")
                    elif adx_value >= self.min_adx and minus_di > plus_di:  # Сильный нисходящий тренд - неправильно!
                        reward -= 3.0
                        # Логи отключены: print(f"[REWARD] ⚠️ Штраф за LONG при нисходящем тренде: -3.0 (ADX={adx_value:.1f}, -DI={minus_di:.1f} > +DI={plus_di:.1f})")
                elif self.position == -1:  # SHORT позиция - УСИЛЕННЫЕ БОНУСЫ
                    if adx_value >= self.min_adx and minus_di > plus_di:  # Сильный нисходящий тренд - правильно!
                        reward += 6.0  # УВЕЛИЧЕНО с 4.0 до 6.0 для стимулирования SHORT
                        # Логи отключены: print(f"[REWARD] ✅ Бонус за SHORT при сильном нисходящем тренде: +6.0 (ADX={adx_value:.1f}, -DI={minus_di:.1f} > +DI={plus_di:.1f})")
                    elif adx_value >= 20 and minus_di > plus_di:  # Умеренный нисходящий тренд - тоже хорошо
                        reward += 3.0
                        # Логи отключены: print(f"[REWARD] ✅ Бонус за SHORT при умеренном нисходящем тренде: +3.0 (ADX={adx_value:.1f})")
                    elif adx_value >= self.min_adx and plus_di > minus_di:  # Сильный восходящий тренд - неправильно!
                        reward -= 3.0
                        # Логи отключены: print(f"[REWARD] ⚠️ Штраф за SHORT при восходящем тренде: -3.0 (ADX={adx_value:.1f}, +DI={plus_di:.1f} > -DI={minus_di:.1f})")
                
                # Бонус за открытие в отличных условиях (используем ADX вместо trend_strength)
                if adx_value >= 30 and volume_ratio > 1.2 and volatility_ratio >= self.min_volatility_ratio and volatility_ratio < 1.6:
                    reward += 5.0  # БОНУС за открытие в отличных условиях (очень сильный тренд)
                    # Логи отключены: print(f"[REWARD] ✅ Бонус за открытие в отличных условиях: +5.0 (ADX={adx_value:.1f}, vol={volume_ratio:.3f}, vol_ratio={volatility_ratio:.4f})")
                elif adx_value >= self.min_adx and volume_ratio > 1.0:
                    reward += 2.0  # Меньший бонус за хорошие условия
                    # Логи отключены: print(f"[REWARD] ✅ Бонус за открытие в нормальных условиях: +2.0 (ADX={adx_value:.1f})")
            except (IndexError, KeyError):
                pass  # Если нет данных, пропускаем
        
        # ШТРАФ ЗА ПРОПУСК ХОРОШИХ ВОЗМОЖНОСТЕЙ (КРИТИЧНО ДЛЯ АКТИВНОСТИ)
        if action == 0 and self.position == 0:  # HOLD без позиции
            try:
                current_atr = float(self.df.iloc[self.current_step]['atr'])
                can_enter = self._check_entry_filters_strict(current_price, current_atr)
                
                if can_enter:
                    row = self.df.iloc[self.current_step]
                    # ЗАМЕНА trend_bias_1h на ADX
                    adx_value = row.get('adx', 0)
                    volume_ratio = row.get('volume_ratio', 1.0)
                    volatility_ratio = row.get('volatility_ratio', 1.5)
                    
                    # Штраф за пропуск отличной возможности (используем ADX вместо trend_strength)
                    if adx_value >= 30 and volume_ratio > 1.2 and volatility_ratio >= self.min_volatility_ratio and volatility_ratio < 1.6:
                        reward -= 3.0  # Штраф за пропуск отличной возможности (очень сильный тренд)
                        # Логи отключены: print(f"[REWARD] ⚠️ Штраф за пропуск отличной возможности: -3.0 (ADX={adx_value:.1f}, vol={volume_ratio:.3f}, vol_ratio={volatility_ratio:.4f})")
                    elif adx_value >= self.min_adx and volume_ratio > 1.0:
                        reward -= 1.0  # Меньший штраф за хорошую возможность
                        # Логи отключены: print(f"[REWARD] ⚠️ Штраф за пропуск хорошей возможности: -1.0 (ADX={adx_value:.1f})")
            except (IndexError, KeyError):
                pass  # Если нет данных, пропускаем
        
        # БОЛЬШАЯ НАГРАДА ЗА TP С ХОРОШИМ RR
        if partial_close:
            if self.partial_closes:
                last_close = self.partial_closes[-1]
                tp_level = last_close['tp_level']
                pnl_ratio = last_close['pnl_ratio']
                
                # УЛУЧШЕННАЯ награда за достижение TP (увеличена)
                tp_bonus = 18.0 * (tp_level * 0.9)  # УВЕЛИЧЕНО с 12.0 до 18.0
                pnl_bonus = min(30.0, pnl_ratio * 400)  # УВЕЛИЧЕНО с 20.0 до 30.0
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