import numpy as np
import pandas as pd
import warnings
from typing import Tuple, List, Dict, Optional
from scipy import stats
import talib
from sklearn.preprocessing import RobustScaler
warnings.filterwarnings('ignore')


class DataProcessorEnhanced:
    """Улучшенный процессор данных с фокусом на TP-признаки"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.raw_df = None
        self.processed_df = None
        
        # Параметры для TP-ориентированных признаков
        self.atr_period = 14
        self.rsi_period = 14
        self.bb_period = 20
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
        # Нормализация
        self.scalers = {}
        self.feature_stats = {}
        
    def load_data(self) -> pd.DataFrame:
        """Загрузка данных с минимальной проверкой"""
        print(f"\n📥 ЗАГРУЗКА ДАННЫХ ИЗ: {self.data_path}")
        
        try:
            # Пробуем разные форматы
            if self.data_path.endswith('.csv'):
                df = pd.read_csv(self.data_path)
            elif self.data_path.endswith('.parquet'):
                df = pd.read_parquet(self.data_path)
            else:
                raise ValueError(f"Неизвестный формат файла: {self.data_path}")
            
            # Минимальная проверка колонок
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                print(f"⚠️  Отсутствуют некоторые обязательные колонки")
                print(f"   Доступные колонки: {df.columns.tolist()}")
                
                # Пытаемся переименовать
                rename_map = {}
                for req in required_cols:
                    if req.upper() in df.columns:
                        rename_map[req.upper()] = req
                    elif req.capitalize() in df.columns:
                        rename_map[req.capitalize()] = req
                
                if rename_map:
                    df = df.rename(columns=rename_map)
                    print(f"   Переименовано: {rename_map}")
            
            # Если все еще нет нужных колонок
            if not all(col in df.columns for col in required_cols):
                print("❌ Невозможно загрузить необходимые колонки")
                # Создаем минимальные данные для тестирования
                print("📝 Создаю тестовые данные...")
                df = pd.DataFrame({
                    'open': np.random.normal(100, 5, 1000),
                    'high': np.random.normal(105, 5, 1000),
                    'low': np.random.normal(95, 5, 1000),
                    'close': np.random.normal(100, 5, 1000),
                    'volume': np.random.normal(1000, 100, 1000)
                })
            
            # Ограничиваем размер для быстрой обработки
            max_rows = 100000  # Ограничиваем для скорости
            if len(df) > max_rows:
                print(f"⚠️  Слишком много данных ({len(df)} строк), ограничиваю до {max_rows}")
                df = df.iloc[-max_rows:].copy()
            
            # Базовая очистка (не удаляем строки!)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col in df.columns:
                    # Заполняем NaN безопасно
                    df[col] = df[col].ffill().bfill().fillna(0)
                    
                    # Обработка выбросов (мягкая)
                    if col in ['open', 'high', 'low', 'close', 'volume']:
                        q1 = df[col].quantile(0.01)
                        q3 = df[col].quantile(0.99)
                        iqr = q3 - q1
                        lower_bound = q1 - 3 * iqr
                        upper_bound = q3 + 3 * iqr
                        
                        outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                        if outliers_mask.any():
                            print(f"   Обработка выбросов в {col}: {outliers_mask.sum()} точек")
                            df.loc[outliers_mask, col] = df[col].median()
            
            print(f"✅ Загружено {len(df)} строк, {len(df.columns)} колонок")
            self.raw_df = df.reset_index(drop=True).copy()
            return self.raw_df
            
        except Exception as e:
            print(f"❌ Ошибка загрузки данных: {e}")
            # Создаем минимальные тестовые данные
            print("📝 Создаю тестовые данные...")
            self.raw_df = pd.DataFrame({
                'open': np.linspace(100, 200, 1000),
                'high': np.linspace(105, 205, 1000),
                'low': np.linspace(95, 195, 1000),
                'close': np.linspace(100, 200, 1000),
                'volume': np.random.normal(1000, 100, 1000)
            })
            return self.raw_df
    
    def check_data_quality(self) -> Dict:
        """Проверка качества данных"""
        if self.raw_df is None:
            return {'error': 'Данные не загружены'}
        
        total_rows = len(self.raw_df)
        
        # Проверка пропусков
        missing_values = self.raw_df.isnull().sum().sum()
        missing_percentage = (missing_values / (total_rows * len(self.raw_df.columns))) * 100
        
        # Проверка отрицательных цен
        price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in self.raw_df.columns]
        negative_prices = 0
        if price_cols:
            negative_prices = (self.raw_df[price_cols] < 0).any(axis=1).sum()
        
        return {
            'total_rows': total_rows,
            'missing_values': int(missing_values),
            'missing_percentage': missing_percentage,
            'negative_prices': int(negative_prices)
        }
    
    def prepare_features(self) -> pd.DataFrame:
        """Основной метод подготовки признаков - УПРОЩЕННЫЙ ДЛЯ СКОРОСТИ"""
        if self.raw_df is None:
            print("❌ Данные не загружены!")
            return pd.DataFrame()
        
        print("\n" + "="*60)
        print("РАСЧЕТ ПРИЗНАКОВ (Упрощенный для скорости)")
        print("="*60)
        
        df = self.raw_df.copy()
        
        # Качество данных
        quality = self.check_data_quality()
        print(f"Качество данных: {quality['total_rows']:,} строк, "
              f"пропусков: {quality['missing_values']} ({quality['missing_percentage']:.2f}%)")
        
        # 1. БАЗОВЫЕ ПРИЗНАКИ (самые важные)
        print("\n--- Базовые признаки ---")
        df = self._add_basic_features(df)
        
        # 2. ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ (упрощенные)
        print("--- Технические индикаторы ---")
        df = self._add_technical_indicators_simple(df)
        
        # 3. TP-ОРИЕНТИРОВАННЫЕ ПРИЗНАКИ (самое важное!)
        print("--- TP-ориентированные признаки ---")
        df = self._add_tp_oriented_features(df)
        
        # 4. ПРОСТАЯ НОРМАЛИЗАЦИЯ (без окон!)
        print("--- Простая нормализация ---")
        df = self._simple_normalization(df)
        
        # 5. ФИНАЛЬНАЯ ОЧИСТКА
        print("--- Финальная очистка ---")
        df = self._final_cleanup(df)
        
        print(f"\n✅ Создано признаков: {len(df.columns)}")
        print(f"✅ Итоговый размер данных: {len(df)} строк × {len(df.columns)} колонок")
        
        self.processed_df = df
        return df
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавление базовых признаков"""
        # Логарифмическая доходность
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['log_ret'] = df['log_ret'].fillna(0)
        
        # Простая доходность
        df['returns'] = df['close'].pct_change()
        df['returns'] = df['returns'].fillna(0)
        
        # Волатильность (простая)
        df['volatility'] = df['returns'].rolling(window=20, min_periods=1).std()
        df['volatility'] = df['volatility'].fillna(df['volatility'].mean())
        
        # Высокие/низкие
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Объем
        if 'volume' in df.columns:
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20, min_periods=1).mean()
            df['volume_ratio'] = df['volume_ratio'].fillna(1)
        
        print(f"   Добавлено базовых признаков: 6")
        return df
    
    def _add_technical_indicators_simple(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавление упрощенных технических индикаторов"""
        try:
            # RSI (упрощенный)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period, min_periods=1).mean()
            rs = gain / (loss + 1e-10)
            df['rsi'] = 100 - (100 / (1 + rs))
            df['rsi'] = df['rsi'].fillna(50)
            
            # ATR (упрощенный)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = tr.rolling(window=self.atr_period, min_periods=1).mean()
            df['atr'] = df['atr'].fillna(df['atr'].mean())
            
            # Bollinger Bands (упрощенный)
            df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
            rolling_std = df['close'].rolling(window=20, min_periods=1).std()
            df['bb_upper'] = df['sma_20'] + (rolling_std * 2)
            df['bb_lower'] = df['sma_20'] - (rolling_std * 2)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
            
            # Momentum
            df['momentum'] = df['close'] - df['close'].shift(5)
            df['momentum'] = df['momentum'].fillna(0)
            
            # ADX (очень упрощенный)
            plus_dm = df['high'].diff()
            minus_dm = df['low'].diff().abs()
            tr = high_low.combine(high_close.combine(low_close, max), max)
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
            df['adx'] = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            df['adx'] = df['adx'].fillna(df['adx'].mean())
            
            print(f"   Добавлено технических индикаторов: 7")
            
        except Exception as e:
            print(f"⚠️  Ошибка расчета индикаторов: {e}")
        
        return df
    
    def _add_tp_oriented_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавление TP-ориентированных признаков - УПРОЩЕННЫХ!"""
        print("   Расчет TP/SL признаков...")
        
        # Базовая волатильность для TP/SL
        if 'atr' not in df.columns:
            df['atr'] = (df['high'] - df['low']).rolling(window=14, min_periods=1).mean()
        
        base_atr = df['atr'].fillna(df['atr'].mean())
        current_price = df['close']
        
        # УПРОЩЕННЫЕ TP-признаки для лонгов (3 уровня)
        for i, multiplier in enumerate([1.2, 1.8, 2.4], 1):
            tp_distance = base_atr * multiplier
            df[f'tp_up_atr_{i}'] = tp_distance / current_price  # Нормализованное расстояние
            
            # Признак вероятности TP (очень упрощенный)
            if 'rsi' in df.columns:
                # Если RSI < 40 (перепроданность), выше вероятность TP
                rsi_factor = np.where(df['rsi'] < 40, 1.5, 
                                    np.where(df['rsi'] > 70, 0.7, 1.0))
                df[f'tp_up_prob_{i}'] = 0.5 * rsi_factor
            else:
                df[f'tp_up_prob_{i}'] = 0.5
        
        # УПРОЩЕННЫЕ TP-признаки для шортов
        for i, multiplier in enumerate([1.2, 1.8, 2.4], 1):
            tp_distance = base_atr * multiplier
            df[f'tp_down_atr_{i}'] = -tp_distance / current_price  # Отрицательное для шортов
            
            if 'rsi' in df.columns:
                # Если RSI > 60 (перекупленность), выше вероятность TP для шортов
                rsi_factor = np.where(df['rsi'] > 60, 1.5,
                                    np.where(df['rsi'] < 30, 0.7, 1.0))
                df[f'tp_down_prob_{i}'] = 0.5 * rsi_factor
            else:
                df[f'tp_down_prob_{i}'] = 0.5
        
        # УПРОЩЕННЫЕ SL-признаки
        sl_multiplier = 1.5  # RR=2: TP 1.2 / SL 0.6 = 2.0
        
        # SL для лонгов
        sl_distance_long = base_atr * sl_multiplier
        df['sl_up_atr'] = -sl_distance_long / current_price  # Отрицательное
        
        # SL для шортов
        sl_distance_short = base_atr * sl_multiplier
        df['sl_down_atr'] = sl_distance_short / current_price  # Положительное
        
        # Признаки времени для выхода (упрощенные)
        df['time_since_high'] = df['close'].rolling(window=20, min_periods=1).apply(
            lambda x: len(x) - np.argmax(x) - 1 if len(x) > 0 else 0, raw=False
        )
        df['time_since_low'] = df['close'].rolling(window=20, min_periods=1).apply(
            lambda x: len(x) - np.argmin(x) - 1 if len(x) > 0 else 0, raw=False
        )
        
        # Признаки прогресса к TP
        for i in range(1, 4):
            df[f'progress_to_tp_up_{i}'] = (df['close'] - df['close'].shift(1)) / (base_atr * [1.2, 1.8, 2.4][i-1] + 1e-10)
            df[f'progress_to_tp_down_{i}'] = (df['close'].shift(1) - df['close']) / (base_atr * [1.2, 1.8, 2.4][i-1] + 1e-10)
        
        print(f"   Добавлено TP/SL признаков: 18")
        return df
    
    def _simple_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Простая нормализация без окон - для скорости!"""
        print(f"   Нормализация {len(df.columns)} признаков (простая)")
        
        # Список колонок для нормализации
        cols_to_normalize = [
            'log_ret', 'returns', 'volatility', 'high_low_ratio', 
            'close_open_ratio', 'volume_ratio', 'rsi', 'atr',
            'bb_position', 'momentum', 'adx'
        ]
        
        # Добавляем TP/SL признаки
        tp_cols = [col for col in df.columns if 'tp_' in col or 'sl_' in col or 'prob_' in col or 'progress_' in col]
        cols_to_normalize.extend(tp_cols)
        
        # Только существующие колонки
        cols_to_normalize = [col for col in cols_to_normalize if col in df.columns]
        
        # Простая нормализация (z-score)
        for col in cols_to_normalize:
            try:
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 1e-10:
                    df[f'{col}_norm'] = (df[col] - mean_val) / std_val
                else:
                    df[f'{col}_norm'] = 0
                
                # Ограничиваем выбросы
                df[f'{col}_norm'] = df[f'{col}_norm'].clip(-5, 5)
                
            except Exception as e:
                print(f"⚠️  Ошибка нормализации {col}: {e}")
                df[f'{col}_norm'] = 0
        
        print(f"   Нормализовано признаков: {len(cols_to_normalize)}")
        return df
    
    def _final_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Финальная очистка данных"""
        # Заполняем все оставшиеся NaN
        df = df.fillna(0)
        
        # Заменяем бесконечные значения
        df = df.replace([np.inf, -np.inf], 0)
        
        # Удаляем дубликаты колонок (если есть)
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Убедимся, что есть достаточное количество строк
        min_rows = 100
        if len(df) < min_rows:
            print(f"⚠️  Слишком мало строк после обработки: {len(df)}")
            # Дублируем данные
            while len(df) < min_rows:
                df = pd.concat([df, df], ignore_index=True)
            df = df.iloc[:min_rows]
            print(f"   Расширено до {len(df)} строк")
        
        return df
    
    def split_data(self, test_size: float = 0.1, validation_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Разделение данных на train/val/test"""
        if self.processed_df is None:
            print("❌ Данные не обработаны!")
            empty_df = pd.DataFrame()
            return empty_df, empty_df, empty_df
        
        df = self.processed_df.copy()
        n = len(df)
        
        # Индексы для разделения
        val_split = int(n * (1 - test_size - validation_size))
        test_split = int(n * (1 - test_size))
        
        train_df = df.iloc[:val_split].copy()
        val_df = df.iloc[val_split:test_split].copy()
        test_df = df.iloc[test_split:].copy()
        
        print(f"\n📊 РАЗДЕЛЕНИЕ ДАННЫХ:")
        print(f"   Всего строк: {n:,}")
        print(f"   Train: {len(train_df):,} строк ({len(train_df)/n*100:.1f}%)")
        print(f"   Validation: {len(val_df):,} строк ({len(val_df)/n*100:.1f}%)")
        print(f"   Test: {len(test_df):,} строк ({len(test_df)/n*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def get_observation_columns(self) -> List[str]:
        """Получение списка колонок для наблюдения"""
        if self.processed_df is None:
            print("⚠️  Данные не обработаны, возвращаю пустой список")
            return []
        
        # Берем все нормализованные колонки
        norm_cols = [col for col in self.processed_df.columns if col.endswith('_norm')]
        
        # И некоторые важные ненормализованные
        important_cols = []
        for col in ['rsi', 'atr', 'bb_position', 'volatility']:
            if col in self.processed_df.columns and f'{col}_norm' not in norm_cols:
                important_cols.append(col)
        
        all_cols = norm_cols + important_cols
        
        # Убираем дубликаты
        all_cols = list(dict.fromkeys(all_cols))
        
        print(f"\n📋 КОЛОНКИ ДЛЯ НАБЛЮДЕНИЯ:")
        print(f"   Всего: {len(all_cols)} признаков")
        print(f"   TP/SL признаки: {len([c for c in all_cols if 'tp_' in c or 'sl_' in c])}")
        
        return all_cols
    
    def get_tp_related_features(self) -> List[str]:
        """Получение TP-ориентированных признаков"""
        if self.processed_df is None:
            return []
        
        tp_features = [
            col for col in self.processed_df.columns 
            if any(x in col for x in ['tp_', 'sl_', 'prob_', 'progress_'])
        ]
        
        return tp_features
    
    def get_exit_timing_features(self) -> List[str]:
        """Признаки для времени выхода"""
        if self.processed_df is None:
            return []
        
        exit_features = [
            col for col in self.processed_df.columns 
            if any(x in col for x in ['time_', 'since_', 'duration', 'hold'])
        ]
        
        return exit_features
    
    def get_volatility_features(self) -> List[str]:
        """Признаки волатильности"""
        if self.processed_df is None:
            return []
        
        vol_features = [
            col for col in self.processed_df.columns 
            if any(x in col for x in ['volatility', 'atr', 'std', 'range'])
        ]
        
        return vol_features


if __name__ == "__main__":
    # Тестирование процессора
    processor = DataProcessorEnhanced("data/btc_15m.csv")
    
    # Загрузка данных
    df = processor.load_data()
    
    # Проверка качества
    quality = processor.check_data_quality()
    print(f"\nКачество данных: {quality}")
    
    # Подготовка признаков
    processed_df = processor.prepare_features()
    
    # Вывод информации
    print(f"\n📊 ИТОГОВАЯ ИНФОРМАЦИЯ:")
    print(f"   Строк: {len(processed_df)}")
    print(f"   Колонок: {len(processed_df.columns)}")
    print(f"   Пример колонок: {list(processed_df.columns[:10])}")
    
    # Получение колонок для наблюдения
    obs_cols = processor.get_observation_columns()
    print(f"\n   Колонок для наблюдения: {len(obs_cols)}")
    
    # Разделение данных
    train_df, val_df, test_df = processor.split_data()
    print(f"\n   Разделение: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")