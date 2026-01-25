import pandas as pd
import numpy as np
import indicators 

class DataProcessor:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.raw_df = None
        self.processed_df = None
        # ДОБАВИТЬ эту строку:
        self.obs_cols = self.get_observation_columns()  # Инициализируем obs_cols

    def load_data(self):
        """Загрузка данных из CSV с нормализацией заголовков"""
        if self.data_path.endswith('.csv'):
            self.raw_df = pd.read_csv(self.data_path)
            self.raw_df.columns = [c.lower() for c in self.raw_df.columns]
        else:
            raise ValueError("Unsupported file format. Use .csv")

    def prepare_features(self):
        """
        v8.8.1 Sniper Elite (Hotfix):
        Исправлена опечатка в вызове функции indicators (breakout_lookback).
        Цель: Восстановить работоспособность процесса обучения.
        """
        if self.raw_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print("--- Расчет индикаторов (v9.3 Sniper Elite) ---")
        # Исправлено: breakback_lookback -> breakout_lookback
        df = indicators.prepare_with_indicators(
            self.raw_df,
            adx_length=14,
            di_length=14,
            sma_length=20,
            rsi_length=14,
            breakout_lookback=20,
            bb_length=20,
            bb_std=2.0,
            atr_length=14,
            ema_fast_length=20,
            ema_slow_length=50,
            ema_timeframe="1h"
        )

        print("--- Глубокая нормализация признаков (v9.3) ---")
        
        # 1. Доходность и Моментум (нормализованный по волатильности)
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        if 'atr' in df.columns:
            df['momentum_std'] = (df['close'] - df['close'].shift(10)) / df['atr']
        
        # 2. Структура рынка и Динамические Уровни
        df['local_high'] = df['high'].rolling(window=50).max()
        df['local_low'] = df['low'].rolling(window=50).min()
        
        df['dist_to_local_high'] = (df['local_high'] - df['close']) / df['close']
        df['dist_to_local_low'] = (df['close'] - df['local_low']) / df['close']
        
        # 3. Волатильность и Относительное Расширение (Range Expansion)
        if 'atr' in df.columns:
            df['atr_norm'] = df['atr'] / df['close']
            # Отношение текущей волатильности к долгосрочной (выявление фаз взрыва)
            df['volatility_ratio'] = df['atr'] / df['atr'].rolling(window=100).mean()
            # Увеличенный потенциал для Reward Shaping
            df['target_dist_up'] = (df['close'] + (df['atr'] * 3.5)) / df['close'] - 1
            df['target_dist_down'] = (df['close'] - (df['atr'] * 3.5)) / df['close'] - 1
        
        # 4. Матрица силы тренда (ADX + DI)
        if 'adx' in df.columns:
            df['adx_norm'] = df['adx'] / 100.0
            # Ускорение ADX: позитивно, если тренд набирает силу
            df['adx_accel'] = df['adx'].diff(3).diff(1) / 5.0
        
        if 'plus_di' in df.columns and 'minus_di' in df.columns:
            df['di_diff'] = (df['plus_di'] - df['minus_di']) / 100.0

        # 5. Старший таймфрейм и Смещение (1h Bias)
        ema_1h_cols = [c for c in df.columns if 'ema' in c and '1h' in c]
        if len(ema_1h_cols) >= 2:
            df['dist_ema_fast_1h'] = (df['close'] - df[ema_1h_cols[0]]) / df['close']
            df['dist_ema_slow_1h'] = (df['close'] - df[ema_1h_cols[1]]) / df['close']
            df['trend_bias_1h'] = np.where(df[ema_1h_cols[0]] > df[ema_1h_cols[1]], 1.0, -1.0)

        # 6. Осцилляторы и Конвергенция
        if 'rsi' in df.columns:
            df['rsi_norm'] = df['rsi'] / 100.0
            # Наклон RSI за 5 свечей (Momentum of Momentum)
            df['rsi_slope'] = df['rsi'].diff(5) / 100.0

        # 7. Объемный анализ
        if 'volume' in df.columns:
            df['vol_rel_sma'] = df['volume'] / df['volume'].rolling(window=20).mean()
            df['vol_spike'] = df['volume'] / df['volume'].rolling(window=100).mean()
        
        if 'vwap' in df.columns:
            df['dist_vwap'] = (df['close'] - df['vwap']) / df['close']

        # 8. Боллинджер и сжатие (Squeeze)
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
            df['dist_bb_upper'] = (df['bb_upper'] - df['close']) / df['close']
            df['dist_bb_lower'] = (df['close'] - df['bb_lower']) / df['close']

        # Финализация признаков
        expected = self.get_observation_columns()
        for col in expected:
            if col not in df.columns:
                df[col] = 0.0

        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        # НОРМАЛИЗАЦИЯ (исправленная)
        df = self.normalize_features(df)
        
        self.processed_df = df

        return self.processed_df
    
    def normalize_features(self, df):
        """Нормализация всех фич к диапазону [-1, 1]"""
        df_normalized = df.copy()
        
        # Получаем список колонок для нормализации
        obs_cols = self.get_observation_columns()
        
        for col in obs_cols:
            if col in df.columns:
                # Проверяем, не нормализована ли уже колонка
                # Некоторые колонки уже в диапазоне [-1, 1]
                already_normalized = any(x in col for x in ['norm', 'ratio', 'diff', 'dist', 'bias', 'slope'])
                
                if not already_normalized:
                    # Z-score нормализация
                    mean = df[col].mean()
                    std = df[col].std()
                    
                    if std > 0:
                        df_normalized[col] = (df[col] - mean) / std
                        # Дополнительное ограничение
                        df_normalized[col] = np.clip(df_normalized[col], -3, 3)
                else:
                    # Уже нормализованные колонки просто ограничиваем
                    df_normalized[col] = np.clip(df[col], -3, 3)
        
        return df_normalized
    
    def split_data(self, test_size=0.1):
        """Разделение на train/test (Timeseries Split)"""
        if self.processed_df is None:
            raise ValueError("Processed data is empty. Call prepare_features() first.")
        
        split_idx = int(len(self.processed_df) * (1 - test_size))
        train_df = self.processed_df.iloc[:split_idx]
        test_df = self.processed_df.iloc[split_idx:]
        return train_df, test_df

    def get_observation_columns(self):
        """
        Признаки v8.8: Добавлены adx_accel и bb_width для лучшего понимания фаз рынка.
        """
        return [
            'log_ret', 'momentum_std', 'rsi_norm', 'rsi_slope', 'adx_norm', 'adx_accel', 'di_diff', 
            'atr_norm', 'volatility_ratio', 'vol_rel_sma', 'vol_spike', 'trend_bias_1h',
            'dist_ema_fast_1h', 'dist_ema_slow_1h',
            'dist_vwap', 'dist_bb_upper', 'dist_bb_lower', 'bb_width',
            'target_dist_up', 'target_dist_down',
            'dist_to_local_high', 'dist_to_local_low'
        ]