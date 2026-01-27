"""
Бэктест стратегий из bot/strategy.py с мультитаймфреймовым анализом.

Тестирует TREND, FLAT и MOMENTUM стратегии на исторических данных
с расчетом винрейта, PnL и других метрик.
"""
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Добавляем путь к проекту для импорта модулей
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot.config import load_settings
from bot.strategy import build_signals, Action, Signal
from bot.indicators import prepare_with_indicators
from bot.simulation import Simulator, Trade


@dataclass
class BacktestMetrics:
    """Метрики бэктеста."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_pct: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    total_signals: int
    long_signals: int
    short_signals: int
    avg_trade_duration_hours: float


@dataclass
class BacktestRecommendation:
    """Рекомендация по улучшению стратегии."""
    category: str  # "risk", "entry", "exit", "filter", "parameter"
    priority: str  # "high", "medium", "low"
    message: str
    suggestion: str


def calculate_metrics(trades: List[Trade], initial_balance: float, signals: List[Signal]) -> BacktestMetrics:
    """Рассчитывает метрики бэктеста."""
    if not trades:
        return BacktestMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            total_pnl_pct=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
            max_drawdown=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            total_signals=len(signals),
            long_signals=len([s for s in signals if s.action == Action.LONG]),
            short_signals=len([s for s in signals if s.action == Action.SHORT]),
            avg_trade_duration_hours=0.0,
        )
    
    winning_trades = [t for t in trades if t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl < 0]
    
    total_pnl = sum(t.pnl for t in trades)
    total_pnl_pct = (total_pnl / initial_balance) * 100 if initial_balance > 0 else 0.0
    
    avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0.0
    avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0.0
    
    total_wins = sum(t.pnl for t in winning_trades)
    total_losses = abs(sum(t.pnl for t in losing_trades))
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0.0
    
    # Расчет максимальной просадки
    cumulative_pnl = np.cumsum([t.pnl for t in trades])
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdown = running_max - cumulative_pnl
    max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
    max_drawdown_pct = (max_drawdown / initial_balance) * 100 if initial_balance > 0 else 0.0
    
    # Расчет Sharpe Ratio (упрощенный)
    if len(trades) > 1:
        returns = [t.pnl / initial_balance for t in trades]
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0.0  # Годовой Sharpe
    else:
        sharpe_ratio = 0.0
    
    # Средняя длительность сделки
    durations = []
    for t in trades:
        if t.entry_time and t.exit_time:
            duration = (t.exit_time - t.entry_time).total_seconds() / 3600  # в часах
            durations.append(duration)
    avg_duration = np.mean(durations) if durations else 0.0
    
    return BacktestMetrics(
        total_trades=len(trades),
        winning_trades=len(winning_trades),
        losing_trades=len(losing_trades),
        win_rate=(len(winning_trades) / len(trades)) * 100 if trades else 0.0,
        total_pnl=total_pnl,
        total_pnl_pct=total_pnl_pct,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        max_drawdown=max_drawdown,
        max_drawdown_pct=max_drawdown_pct,
        sharpe_ratio=sharpe_ratio,
        total_signals=len(signals),
        long_signals=len([s for s in signals if s.action == Action.LONG]),
        short_signals=len([s for s in signals if s.action == Action.SHORT]),
        avg_trade_duration_hours=avg_duration,
    )


def generate_recommendations(metrics: BacktestMetrics, trades: List[Trade], strategy_type: str = "trend") -> List[BacktestRecommendation]:
    """Генерирует рекомендации по улучшению стратегии."""
    recommendations = []
    
    # Анализ винрейта
    if metrics.win_rate < 40:
        recommendations.append(BacktestRecommendation(
            category="entry",
            priority="high",
            message=f"Низкий винрейт: {metrics.win_rate:.1f}%",
            suggestion=f"Рассмотрите ужесточение фильтров входа (ADX > 25, RSI экстремумы, объем). Для {strategy_type.upper()} стратегии проверьте качество сигналов pullback/breakout. Возможно, стоит увеличить пороги индикаторов для более сильных сигналов."
        ))
    elif metrics.win_rate > 60:
        recommendations.append(BacktestRecommendation(
            category="entry",
            priority="low",
            message=f"Высокий винрейт: {metrics.win_rate:.1f}%",
            suggestion="Хороший винрейт. Рассмотрите возможность увеличения размера позиций или уменьшения SL для увеличения прибыли."
        ))
    
    # Анализ Profit Factor
    if metrics.profit_factor < 1.0:
        recommendations.append(BacktestRecommendation(
            category="risk",
            priority="high",
            message=f"Profit Factor ниже 1.0: {metrics.profit_factor:.2f}",
            suggestion=f"Стратегия убыточна. Пересмотрите логику входа и выхода для {strategy_type.upper()} стратегии. Возможно, стоит увеличить TP/SL соотношение или улучшить фильтры входа. Проверьте MTF фильтры - возможно они слишком строгие или наоборот недостаточно фильтруют."
        ))
    elif metrics.profit_factor < 1.5:
        recommendations.append(BacktestRecommendation(
            category="risk",
            priority="medium",
            message=f"Profit Factor низкий: {metrics.profit_factor:.2f}",
            suggestion="Рассмотрите оптимизацию соотношения TP/SL. Увеличьте TP или уменьшите SL для улучшения соотношения риск/прибыль."
        ))
    
    # Анализ максимальной просадки
    if metrics.max_drawdown_pct > 30:
        recommendations.append(BacktestRecommendation(
            category="risk",
            priority="high",
            message=f"Большая просадка: {metrics.max_drawdown_pct:.1f}%",
            suggestion="Уменьшите размер позиций или увеличьте диверсификацию. Рассмотрите добавление фильтров для избежания торговли в неблагоприятных условиях (например, фильтр по волатильности или тренду)."
        ))
    elif metrics.max_drawdown_pct > 15:
        recommendations.append(BacktestRecommendation(
            category="risk",
            priority="medium",
            message=f"Умеренная просадка: {metrics.max_drawdown_pct:.1f}%",
            suggestion="Просадка в допустимых пределах, но можно улучшить. Рассмотрите добавление механизма остановки торговли после серии убытков."
        ))
    
    # Анализ средних прибылей и убытков
    if metrics.avg_loss != 0 and abs(metrics.avg_win / metrics.avg_loss) < 1.5:
        recommendations.append(BacktestRecommendation(
            category="exit",
            priority="medium",
            message=f"Соотношение средних прибыль/убыток низкое: {abs(metrics.avg_win / metrics.avg_loss):.2f}",
            suggestion="Увеличьте Take Profit или уменьшите Stop Loss. Рассмотрите использование трейлинг-стопа для защиты прибыли. Для трендовых стратегий важно ловить большие движения."
        ))
    
    # Анализ Sharpe Ratio
    if metrics.sharpe_ratio < 0:
        recommendations.append(BacktestRecommendation(
            category="risk",
            priority="high",
            message=f"Отрицательный Sharpe Ratio: {metrics.sharpe_ratio:.2f}",
            suggestion="Стратегия имеет плохое соотношение риск/доходность. Необходимо пересмотреть логику входа/выхода или фильтры. Возможно, стоит добавить дополнительные фильтры по волатильности или тренду."
        ))
    elif metrics.sharpe_ratio < 1.0:
        recommendations.append(BacktestRecommendation(
            category="risk",
            priority="medium",
            message=f"Sharpe Ratio ниже оптимального: {metrics.sharpe_ratio:.2f}",
            suggestion="Рассмотрите оптимизацию стратегии для улучшения соотношения риск/доходность. Хороший Sharpe Ratio должен быть > 1.0."
        ))
    
    # Анализ количества сигналов
    if metrics.total_signals == 0:
        recommendations.append(BacktestRecommendation(
            category="parameter",
            priority="high",
            message="Нет сигналов",
            suggestion=f"Стратегия {strategy_type.upper()} не генерирует сигналы. Проверьте параметры индикаторов (ADX, RSI, SMA). Возможно, условия слишком строгие или MTF фильтры блокируют все сигналы."
        ))
    elif metrics.total_signals < 10:
        recommendations.append(BacktestRecommendation(
            category="parameter",
            priority="low",
            message=f"Мало сигналов: {metrics.total_signals}",
            suggestion="Рассмотрите смягчение условий входа для увеличения количества сделок, если это не ухудшит качество. Проверьте MTF фильтры - возможно они слишком строгие."
        ))
    
    # Анализ длительности сделок
    if metrics.avg_trade_duration_hours > 48:
        recommendations.append(BacktestRecommendation(
            category="exit",
            priority="low",
            message=f"Долгие сделки: {metrics.avg_trade_duration_hours:.1f} часов",
            suggestion="Рассмотрите более агрессивные условия выхода. Возможно, стоит уменьшить TP или добавить временной фильтр. Для трендовых стратегий это может быть нормально, если сделки прибыльные."
        ))
    
    # Анализ соотношения LONG/SHORT сигналов
    if metrics.total_signals > 0:
        long_ratio = metrics.long_signals / metrics.total_signals
        if long_ratio > 0.7 or long_ratio < 0.3:
            recommendations.append(BacktestRecommendation(
                category="filter",
                priority="low",
                message=f"Дисбаланс сигналов: LONG {metrics.long_signals} ({long_ratio*100:.1f}%), SHORT {metrics.short_signals} ({(1-long_ratio)*100:.1f}%)",
                suggestion="Сигналы сильно смещены в одну сторону. Проверьте логику определения тренда и фильтры. Возможно, стоит пересмотреть условия для более сбалансированного распределения."
            ))
    
    # Анализ эффективности по типам сигналов (если есть данные)
    if trades:
        pullback_trades = [t for t in trades if "pullback" in (t.entry_reason or "").lower()]
        breakout_trades = [t for t in trades if "breakout" in (t.entry_reason or "").lower()]
        
        if pullback_trades and breakout_trades:
            pullback_wins = len([t for t in pullback_trades if t.pnl > 0])
            breakout_wins = len([t for t in breakout_trades if t.pnl > 0])
            pullback_win_rate = (pullback_wins / len(pullback_trades)) * 100 if pullback_trades else 0.0
            breakout_win_rate = (breakout_wins / len(breakout_trades)) * 100 if breakout_trades else 0.0
            
            if abs(pullback_win_rate - breakout_win_rate) > 20:
                worse_type = "pullback" if pullback_win_rate < breakout_win_rate else "breakout"
                recommendations.append(BacktestRecommendation(
                    category="entry",
                    priority="medium",
                    message=f"Разница в эффективности типов сигналов: Pullback {pullback_win_rate:.1f}% vs Breakout {breakout_win_rate:.1f}%",
                    suggestion=f"Сигналы типа '{worse_type}' показывают значительно худшие результаты. Рассмотрите ужесточение фильтров для этого типа или временное отключение менее эффективных сигналов."
                ))
    
    return recommendations


def load_timeframe_data(base_path: str, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    """
    Загружает данные с указанного таймфрейма из CSV файлов.
    Поддерживает разные форматы имен файлов (btcusdt_15m.csv, btc_15m.csv).
    
    Args:
        base_path: Базовый путь к папке data
        symbol: Торговая пара (BTCUSDT, ETHUSDT, SOLUSDT)
        timeframe: Таймфрейм ('15m', '1h', '4h')
    
    Returns:
        DataFrame с данными или None если файл не найден
    """
    # Определяем возможные имена файлов (поддержка разных форматов)
    symbol_lower = symbol.lower()
    possible_filenames = [
        f"{symbol_lower}_{timeframe}.csv",  # btcusdt_15m.csv
        f"{symbol_lower[:3]}_{timeframe}.csv",  # btc_15m.csv (fallback)
    ]
    
    # Пробуем найти файл
    filepath = None
    for filename in possible_filenames:
        test_path = os.path.join(base_path, filename)
        if os.path.exists(test_path):
            filepath = test_path
            break
    
    if filepath is None:
        return None
    
    try:
        df = pd.read_csv(filepath)
        
        # Подготовка данных
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
        else:
            return None
        
        # Убеждаемся что есть нужные колонки
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            return None
        
        return df[required_cols]
    except Exception as e:
        print(f"   ⚠️ Error loading {timeframe} data: {e}")
        return None


def run_strategy_backtest(
    csv_path: str,
    strategy_type: str = "trend",  # "trend", "flat", "momentum"
    use_mtf_filter: bool = True,
    mtf_timeframe: str = "1h",
    initial_balance: float = 1000.0,
    symbol: str = "BTCUSDT",
    use_all_timeframes: bool = True,  # Использовать все доступные таймфреймы
    verbose: bool = True,  # Выводить детальную информацию
    days: Optional[int] = None,  # Ограничить данные последними N днями (None = все данные)
) -> Dict[str, Any]:
    """
    Запускает бэктест стратегии на исторических данных.
    
    Args:
        csv_path: Путь к CSV файлу с историческими данными
        strategy_type: Тип стратегии ("trend", "flat", "momentum")
        use_mtf_filter: Использовать ли мультитаймфреймовый фильтр
        mtf_timeframe: Таймфрейм для MTF анализа ("1h", "4h")
        initial_balance: Начальный баланс
        symbol: Торговая пара
        days: Ограничить данные последними N днями (None = использовать все данные)
    
    Returns:
        Словарь с результатами бэктеста
    """
    print("=" * 80)
    print(f"📊 BACKTEST: {strategy_type.upper()} Strategy {'with MTF' if use_mtf_filter else 'without MTF'}")
    print("=" * 80)
    
    # Загрузка данных из локального CSV файла (без обращения к API)
    print(f"\n📁 Loading data from local CSV: {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Подготовка данных
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
    else:
        raise ValueError("CSV must have 'datetime' or 'timestamp' column")
    
    # Ограничиваем данные последними N днями если указано
    if days is not None:
        if days < 1 or days > 30:
            raise ValueError(f"days must be between 1 and 30, got {days}")
        last_date = df.index[-1]
        start_date = last_date - pd.Timedelta(days=days)
        df = df[df.index >= start_date]
        if verbose:
            print(f"   ⏱️ Limited to last {days} days")
    
    if verbose:
        print(f"   Loaded {len(df)} candles")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    
    # Загрузка данных с высших таймфреймов для MTF анализа
    df_1h = None
    df_4h = None
    if use_mtf_filter and use_all_timeframes:
        if verbose:
            print(f"\n📊 Loading higher timeframe data for MTF analysis...")
        data_dir = os.path.dirname(csv_path) or "data"
        
        df_1h = load_timeframe_data(data_dir, symbol, "1h")
        if df_1h is not None and days is not None:
            # Ограничиваем данные высших таймфреймов тоже
            last_date = df_1h.index[-1]
            start_date = last_date - pd.Timedelta(days=days)
            df_1h = df_1h[df_1h.index >= start_date]
        if verbose:
            if df_1h is not None:
                print(f"   ✅ Loaded 1H data: {len(df_1h)} candles ({df_1h.index[0]} to {df_1h.index[-1]})")
            else:
                print(f"   ⚠️ 1H data not found, will resample from 15m")
        
        df_4h = load_timeframe_data(data_dir, symbol, "4h")
        if df_4h is not None and days is not None:
            # Ограничиваем данные высших таймфреймов тоже
            last_date = df_4h.index[-1]
            start_date = last_date - pd.Timedelta(days=days)
            df_4h = df_4h[df_4h.index >= start_date]
        if verbose:
            if df_4h is not None:
                print(f"   ✅ Loaded 4H data: {len(df_4h)} candles ({df_4h.index[0]} to {df_4h.index[-1]})")
            else:
                print(f"   ⚠️ 4H data not found, will resample from 15m")
    
    # Подготовка индикаторов
    if verbose:
        print(f"\n🔧 Preparing indicators...")
    settings = load_settings()
    # Подготавливаем индикаторы с использованием готовых данных высших таймфреймов
    df_ready = prepare_with_indicators(
        df,
        adx_length=settings.strategy.adx_length,
        di_length=settings.strategy.di_length,
        sma_length=settings.strategy.sma_length,
        rsi_length=settings.strategy.rsi_length,
        breakout_lookback=settings.strategy.breakout_lookback,
        bb_length=settings.strategy.bb_length,
        bb_std=settings.strategy.bb_std,
        atr_length=14,
        ema_fast_length=settings.strategy.ema_fast_length,
        df_1h=df_1h if use_all_timeframes else None,  # Передаем готовые данные 1H
        df_4h=df_4h if use_all_timeframes else None,  # Передаем готовые данные 4H
        ema_slow_length=settings.strategy.ema_slow_length,
    )
    
    # Генерация сигналов
    if verbose:
        print(f"\n📈 Generating {strategy_type.upper()} signals...")
    use_momentum = (strategy_type == "momentum")
    
    # Подготавливаем параметры для стратегии
    strategy_params = {
        'sma_period': getattr(settings.strategy, 'sma_length', 21),
        'atr_period': 14,
        'atr_multiplier': 2.0,
        'max_pyramid': 2,
        'min_history': 100,
    }
    
    # Подготавливаем state с данными высших таймфреймов
    strategy_state = {}
    if use_mtf_filter and use_all_timeframes:
        if df_1h is not None:
            strategy_state['df_1h'] = df_1h
        if df_4h is not None:
            strategy_state['df_4h'] = df_4h
    
    if use_mtf_filter:
        strategy_params['use_mtf_filter'] = True
        strategy_params['mtf_timeframe'] = mtf_timeframe
        strategy_params['mtf_ema_period'] = 50
        # Блокируем neutral сигналы для повышения качества входов
        strategy_state['mtf_block_neutral'] = True
        if verbose:
            if use_all_timeframes and (df_1h is not None or df_4h is not None):
                print(f"   Using MTF filter: Multi-timeframe consensus (1H + 4H) - blocking neutral signals")
            else:
                print(f"   Using MTF filter: {mtf_timeframe} timeframe (resampled) - blocking neutral signals")
    else:
        strategy_params['use_mtf_filter'] = False
    
    # Устанавливаем флаг бэктеста в state
    strategy_state['backtest_mode'] = True
    # Отключаем фильтр по времени в бэктесте (чтобы протестировать все часы)
    strategy_state['enable_time_filter'] = False
    # Добавляем символ в state для адаптивных параметров
    strategy_state['symbol'] = symbol
    strategy_state['trading_symbol'] = symbol
    
    print(f"\n   Calling build_signals for backtesting:")
    print(f"      DataFrame shape: {df_ready.shape}")
    print(f"      Strategy: {strategy_type}")
    print(f"      MTF filter: {use_mtf_filter}")
    print(f"      State keys: {list(strategy_state.keys())}")
    print(f"      Iterating over {len(df_ready)} candles...")
    
    # Для бэктеста нужно генерировать сигналы для каждой свечи
    # Итерируемся по DataFrame, начиная с min_history
    signals = []
    min_history = strategy_params.get('min_history', 100)
    
    # Обновляем данные высших таймфреймов для каждой итерации
    for i in range(min_history, len(df_ready)):
        # Берем данные до текущего момента (включительно)
        df_slice = df_ready.iloc[:i+1]
        
        # Обновляем state для текущего момента
        current_state = strategy_state.copy()
        current_state['last_signal_idx'] = -100  # Сбрасываем cooldown для бэктеста
        current_state['symbol'] = symbol  # Убеждаемся что символ есть в state
        current_state['trading_symbol'] = symbol
        
        # Обновляем данные высших таймфреймов до текущего момента
        if 'df_1h' in current_state and current_state['df_1h'] is not None:
            # Находим последнюю свечу 1H до текущего момента
            current_time = df_slice.index[-1]
            df_1h_slice = current_state['df_1h'][current_state['df_1h'].index <= current_time]
            if len(df_1h_slice) > 0:
                current_state['df_1h'] = df_1h_slice
        
        if 'df_4h' in current_state and current_state['df_4h'] is not None:
            # Находим последнюю свечу 4H до текущего момента
            current_time = df_slice.index[-1]
            df_4h_slice = current_state['df_4h'][current_state['df_4h'].index <= current_time]
            if len(df_4h_slice) > 0:
                current_state['df_4h'] = df_4h_slice
        
        # Генерируем сигнал для текущей свечи
        try:
            candle_signals = build_signals(
                df_slice,
                settings.strategy,
                use_momentum=use_momentum,
                use_liquidity=False,
                params=strategy_params,
                state=current_state,
            )
            signals.extend(candle_signals)
        except Exception as e:
            # Пропускаем ошибки для отдельных свечей
            if verbose and i % 1000 == 0:  # Логируем каждую 1000-ю ошибку
                print(f"   Warning: Error at candle {i}: {e}")
            continue
        
        # Прогресс каждые 5000 свечей (только если verbose)
        if verbose and i % 5000 == 0:
            print(f"   Processed {i}/{len(df_ready)} candles, signals so far: {len(signals)}")
        
        # В бэктесте не нужно логировать каждую свечу - слишком много
        # Логирование отключено для производительности
    
    print(f"   build_signals returned {len(signals)} signals total")
    
    # Детальное логирование всех сигналов
    print(f"\n🔍 SIGNAL ANALYSIS for {symbol}:")
    print(f"   Total signals generated: {len(signals)}")
    
    # Анализ сигналов по типам
    trend_signals = [s for s in signals if s.reason.startswith("trend_")]
    range_signals = [s for s in signals if s.reason.startswith("range_")]
    momentum_signals = [s for s in signals if s.reason.startswith("momentum_")]
    hold_signals = [s for s in signals if s.action == Action.HOLD]
    long_signals_all = [s for s in signals if s.action == Action.LONG]
    short_signals_all = [s for s in signals if s.action == Action.SHORT]
    
    print(f"   By type:")
    print(f"      TREND signals: {len(trend_signals)}")
    print(f"      RANGE/FLAT signals: {len(range_signals)}")
    print(f"      MOMENTUM signals: {len(momentum_signals)}")
    print(f"      HOLD signals: {len(hold_signals)}")
    print(f"   By action:")
    print(f"      LONG: {len(long_signals_all)}")
    print(f"      SHORT: {len(short_signals_all)}")
    
    # Анализ причин сигналов
    if signals:
        reason_counts = {}
        for sig in signals:
            reason = sig.reason
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        print(f"   Top reasons:")
        sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for reason, count in sorted_reasons:
            print(f"      {reason}: {count}")
    
    # Проверка индикаторов на последних свечах
    if len(df_ready) > 0:
        last_row = df_ready.iloc[-1]
        print(f"\n   Last candle indicators:")
        print(f"      Price: ${last_row.get('close', 'N/A'):.2f}" if pd.notna(last_row.get('close')) else f"      Price: N/A")
        print(f"      SMA: ${last_row.get('sma', 'N/A'):.2f}" if pd.notna(last_row.get('sma')) else f"      SMA: N/A")
        print(f"      ADX: {last_row.get('adx', 'N/A'):.2f}" if pd.notna(last_row.get('adx')) else f"      ADX: N/A")
        print(f"      RSI: {last_row.get('rsi', 'N/A'):.2f}" if pd.notna(last_row.get('rsi')) else f"      RSI: N/A")
        print(f"      ATR: ${last_row.get('atr', 'N/A'):.2f}" if pd.notna(last_row.get('atr')) else f"      ATR: N/A")
        print(f"      Plus DI: {last_row.get('plus_di', 'N/A'):.2f}" if pd.notna(last_row.get('plus_di')) else f"      Plus DI: N/A")
        print(f"      Minus DI: {last_row.get('minus_di', 'N/A'):.2f}" if pd.notna(last_row.get('minus_di')) else f"      Minus DI: N/A")
        
        # Проверка MTF консенсуса если используется
        if use_mtf_filter and strategy_state:
            print(f"\n   MTF Analysis:")
            if 'df_1h' in strategy_state and strategy_state['df_1h'] is not None:
                df_1h_check = strategy_state['df_1h']
                if len(df_1h_check) > 0:
                    print(f"      1H data available: {len(df_1h_check)} candles")
                    if len(df_1h_check) >= 50:
                        last_1h = df_1h_check.iloc[-1]
                        ema_1h_50 = df_1h_check['close'].ewm(span=50, adjust=False).mean().iloc[-1]
                        print(f"      1H Close: ${last_1h.get('close', 'N/A'):.2f}, EMA50: ${ema_1h_50:.2f}")
                        print(f"      1H Bias: {'BULLISH' if last_1h.get('close', 0) > ema_1h_50 else 'BEARISH'}")
            if 'df_4h' in strategy_state and strategy_state['df_4h'] is not None:
                df_4h_check = strategy_state['df_4h']
                if len(df_4h_check) > 0:
                    print(f"      4H data available: {len(df_4h_check)} candles")
                    if len(df_4h_check) >= 50:
                        last_4h = df_4h_check.iloc[-1]
                        ema_4h_50 = df_4h_check['close'].ewm(span=50, adjust=False).mean().iloc[-1]
                        print(f"      4H Close: ${last_4h.get('close', 'N/A'):.2f}, EMA50: ${ema_4h_50:.2f}")
                        print(f"      4H Bias: {'BULLISH' if last_4h.get('close', 0) > ema_4h_50 else 'BEARISH'}")
    
    # Фильтруем сигналы по типу стратегии
    actionable_signals = []
    if strategy_type == "trend":
        actionable_signals = [s for s in signals if s.reason.startswith("trend_") and s.action in (Action.LONG, Action.SHORT)]
    elif strategy_type == "flat":
        actionable_signals = [s for s in signals if s.reason.startswith("range_") and s.action in (Action.LONG, Action.SHORT)]
    elif strategy_type == "momentum":
        actionable_signals = [s for s in signals if s.reason.startswith("momentum_") and s.action in (Action.LONG, Action.SHORT)]
    
    print(f"\n   After filtering for '{strategy_type}' strategy:")
    print(f"      Actionable signals: {len(actionable_signals)}")
    print(f"      LONG actionable: {len([s for s in actionable_signals if s.action == Action.LONG])}")
    print(f"      SHORT actionable: {len([s for s in actionable_signals if s.action == Action.SHORT])}")
    
    # Анализ причин блокировки
    if len(signals) > 0 and len(actionable_signals) == 0:
        print(f"\n   ⚠️ All signals were filtered out!")
        print(f"   Checking why signals were blocked:")
        
        # Проверяем сигналы которые могли быть заблокированы MTF фильтром
        blocked_by_mtf = []
        for sig in signals:
            if sig.action in (Action.LONG, Action.SHORT):
                if sig.reason.startswith("mtf_filter"):
                    blocked_by_mtf.append(sig)
        
        if blocked_by_mtf:
            print(f"      Blocked by MTF filter: {len(blocked_by_mtf)}")
            for sig in blocked_by_mtf[:5]:  # Показываем первые 5
                print(f"         {sig.action.value} @ ${sig.price:.2f} - {sig.reason}")
        
        # Проверяем сигналы с другими причинами блокировки
        other_blocks = [s for s in signals if s.action == Action.HOLD and s.reason not in ['cooldown', 'no_action']]
        if other_blocks:
            block_reasons = {}
            for sig in other_blocks:
                block_reasons[sig.reason] = block_reasons.get(sig.reason, 0) + 1
            print(f"      Other blocks:")
            for reason, count in sorted(block_reasons.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"         {reason}: {count}")
    
    if verbose:
        print(f"\n   Generated {len(actionable_signals)} actionable signals")
        print(f"   LONG: {len([s for s in actionable_signals if s.action == Action.LONG])}")
        print(f"   SHORT: {len([s for s in actionable_signals if s.action == Action.SHORT])}")
    
    if not actionable_signals:
        print(f"\n   ⚠️ No actionable signals for {symbol}!")
        print(f"   Total signals: {len(signals)}, Filtered out: {len(signals) - len(actionable_signals)}")
        empty_metrics = calculate_metrics([], initial_balance, signals)
        recommendations = generate_recommendations(empty_metrics, [], strategy_type)
        return {
            "metrics": empty_metrics,
            "trades": [],
            "signals": [],
            "recommendations": recommendations,
        }
    
    # Запуск симулятора
    if verbose:
        print(f"\n💰 Running simulator (Initial balance: ${initial_balance:.2f})...")
    simulator = Simulator(settings)
    result = simulator.run(df_ready, actionable_signals)
    
    trades = result["trades"]
    total_pnl = result["total_pnl"]
    
    # Расчет метрик
    metrics = calculate_metrics(trades, initial_balance, actionable_signals)
    
    # Вывод результатов
    if verbose:
        print("\n" + "=" * 80)
        print("📊 BACKTEST RESULTS")
        print("=" * 80)
        print(f"\n📈 Signals:")
        print(f"   Total signals: {metrics.total_signals}")
        print(f"   LONG signals: {metrics.long_signals}")
        print(f"   SHORT signals: {metrics.short_signals}")
        
        # Вывод информации о MTF консенсусе (берем из последнего сигнала для актуальности)
        if use_mtf_filter and actionable_signals:
            mtf_info = {}
            # Проверяем последние сигналы (более актуальные)
            for sig in reversed(actionable_signals[-10:]):
                if hasattr(sig, 'indicators_info') and sig.indicators_info:
                    mtf_consensus = sig.indicators_info.get('mtf_consensus')
                    if mtf_consensus:
                        mtf_info = mtf_consensus
                        break
                    # Также проверяем mtf_bias напрямую
                    mtf_bias = sig.indicators_info.get('mtf_bias')
                    mtf_timeframe = sig.indicators_info.get('mtf_timeframe', '1h')
                    if mtf_bias:
                        if not mtf_info:
                            mtf_info = {}
                        if mtf_timeframe == '1h':
                            mtf_info['1h_bias'] = mtf_bias
                        elif mtf_timeframe == '4h':
                            mtf_info['4h_bias'] = mtf_bias
            
            # Если mtf_info пустой, пытаемся получить из strategy_state
            if not mtf_info and strategy_state:
                if 'df_1h' in strategy_state and strategy_state['df_1h'] is not None:
                    df_1h_check = strategy_state['df_1h']
                    if len(df_1h_check) >= 50:
                        last_1h = df_1h_check.iloc[-1]
                        ema_1h_50 = df_1h_check['close'].ewm(span=50, adjust=False).mean().iloc[-1]
                        mtf_info['1h_bias'] = 'bullish' if last_1h.get('close', 0) > ema_1h_50 else 'bearish'
                if 'df_4h' in strategy_state and strategy_state['df_4h'] is not None:
                    df_4h_check = strategy_state['df_4h']
                    if len(df_4h_check) >= 50:
                        last_4h = df_4h_check.iloc[-1]
                        ema_4h_50 = df_4h_check['close'].ewm(span=50, adjust=False).mean().iloc[-1]
                        mtf_info['4h_bias'] = 'bullish' if last_4h.get('close', 0) > ema_4h_50 else 'bearish'
            
            if mtf_info:
                print(f"\n🌐 Multi-Timeframe Analysis:")
                print(f"   1H Bias: {mtf_info.get('1h_bias', 'N/A')}")
                print(f"   4H Bias: {mtf_info.get('4h_bias', 'N/A')}")
                # Вычисляем консенсус если есть оба bias
                if '1h_bias' in mtf_info and '4h_bias' in mtf_info:
                    if mtf_info['1h_bias'] == mtf_info['4h_bias']:
                        mtf_info['consensus'] = mtf_info['1h_bias']
                    else:
                        mtf_info['consensus'] = 'neutral'
                print(f"   Consensus: {mtf_info.get('consensus', 'N/A')}")
                print(f"   Trend Strength: {mtf_info.get('trend_strength', 0.0):.2f}")
        
        # Анализ эффективности по типам сигналов (pullback vs breakout)
        pullback_trades = []
        breakout_trades = []
        for trade in trades:
            entry_reason = trade.entry_reason or ""
            if "pullback" in entry_reason.lower():
                pullback_trades.append(trade)
            elif "breakout" in entry_reason.lower():
                breakout_trades.append(trade)
        
        if pullback_trades or breakout_trades:
            print(f"\n📊 Signal Type Analysis:")
            if pullback_trades:
                pullback_wins = [t for t in pullback_trades if t.pnl > 0]
                pullback_pnl = sum(t.pnl for t in pullback_trades)
                pullback_win_rate = (len(pullback_wins) / len(pullback_trades)) * 100 if pullback_trades else 0.0
                pullback_avg_win = sum(t.pnl for t in pullback_wins) / len(pullback_wins) if pullback_wins else 0.0
                pullback_avg_loss = sum(t.pnl for t in pullback_trades if t.pnl < 0) / len([t for t in pullback_trades if t.pnl < 0]) if [t for t in pullback_trades if t.pnl < 0] else 0.0
                print(f"   Pullback Signals:")
                print(f"      Trades: {len(pullback_trades)} ({len(pullback_trades)/len(trades)*100:.1f}%)")
                print(f"      Win Rate: {pullback_win_rate:.2f}%")
                print(f"      Total PnL: ${pullback_pnl:.2f}")
                print(f"      Avg Win: ${pullback_avg_win:.2f}, Avg Loss: ${pullback_avg_loss:.2f}")
            
            if breakout_trades:
                breakout_wins = [t for t in breakout_trades if t.pnl > 0]
                breakout_pnl = sum(t.pnl for t in breakout_trades)
                breakout_win_rate = (len(breakout_wins) / len(breakout_trades)) * 100 if breakout_trades else 0.0
                breakout_avg_win = sum(t.pnl for t in breakout_wins) / len(breakout_wins) if breakout_wins else 0.0
                breakout_avg_loss = sum(t.pnl for t in breakout_trades if t.pnl < 0) / len([t for t in breakout_trades if t.pnl < 0]) if [t for t in breakout_trades if t.pnl < 0] else 0.0
                print(f"   Breakout Signals:")
                print(f"      Trades: {len(breakout_trades)} ({len(breakout_trades)/len(trades)*100:.1f}%)")
                print(f"      Win Rate: {breakout_win_rate:.2f}%")
                print(f"      Total PnL: ${breakout_pnl:.2f}")
                print(f"      Avg Win: ${breakout_avg_win:.2f}, Avg Loss: ${breakout_avg_loss:.2f}")
        
        print(f"\n💼 Trades:")
        print(f"   Total trades: {metrics.total_trades}")
        print(f"   Winning trades: {metrics.winning_trades}")
        print(f"   Losing trades: {metrics.losing_trades}")
        print(f"   Win Rate: {metrics.win_rate:.2f}%")
        
        print(f"\n💰 PnL:")
        print(f"   Total PnL: ${metrics.total_pnl:.2f} ({metrics.total_pnl_pct:+.2f}%)")
        print(f"   Average Win: ${metrics.avg_win:.2f}")
        print(f"   Average Loss: ${metrics.avg_loss:.2f}")
        print(f"   Profit Factor: {metrics.profit_factor:.2f}")
        
        print(f"\n📉 Risk Metrics:")
        print(f"   Max Drawdown: ${metrics.max_drawdown:.2f} ({metrics.max_drawdown_pct:.2f}%)")
        print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"   Avg Trade Duration: {metrics.avg_trade_duration_hours:.1f} hours")
        
        # Генерируем рекомендации (будем использовать их позже для сохранения)
        recommendations = generate_recommendations(metrics, trades, strategy_type)
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in recommendations:
                priority_icon = "🔴" if rec.priority == "high" else "🟡" if rec.priority == "medium" else "🟢"
                print(f"   {priority_icon} [{rec.priority.upper()}] {rec.category.upper()}")
                print(f"      Issue: {rec.message}")
                print(f"      Suggestion: {rec.suggestion}")
    else:
        # Компактный вывод для мультисимвольного режима - генерируем рекомендации для возврата
        recommendations = generate_recommendations(metrics, trades, strategy_type)
    print("\n" + "=" * 80)
    
    # Сохранение результатов в CSV
    if trades:
        trades_df = pd.DataFrame([
            {
                "entry_time": t.entry_time.isoformat() if t.entry_time else "",
                "exit_time": t.exit_time.isoformat() if t.exit_time else "",
                "side": t.side.value,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "size_usd": t.size_usd,
                "pnl": t.pnl,
                "pnl_pct": (t.pnl / initial_balance) * 100,
                "entry_reason": t.entry_reason or "",
                "exit_reason": t.exit_reason,
            }
            for t in trades
        ])
        
        output_file = f"backtest_{strategy_type}_{symbol}_{'mtf' if use_mtf_filter else 'no_mtf'}.csv"
        trades_df.to_csv(output_file, index=False)
        print(f"\n💾 Trades saved to: {output_file}")
    
    # Сохраняем рекомендации в файл
    if recommendations:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rec_file = f"results/{strategy_type}_backtest_recommendations_{timestamp}.txt"
        os.makedirs("results", exist_ok=True)
        with open(rec_file, "w", encoding="utf-8") as f:
            f.write(f"{strategy_type.upper()} STRATEGY BACKTEST RECOMMENDATIONS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Symbol: {symbol}\n")
            f.write(f"Strategy: {strategy_type.upper()}\n")
            f.write(f"MTF Filter: {'ON' if use_mtf_filter else 'OFF'}\n")
            f.write(f"Total Trades: {metrics.total_trades}\n")
            f.write(f"Win Rate: {metrics.win_rate:.2f}%\n")
            f.write(f"Total PnL: ${metrics.total_pnl:.2f}\n")
            f.write(f"Profit Factor: {metrics.profit_factor:.2f}\n\n")
            f.write("RECOMMENDATIONS:\n")
            f.write("=" * 80 + "\n\n")
            for rec in recommendations:
                f.write(f"[{symbol}] [{rec.priority.upper()}] {rec.category.upper()}\n")
                f.write(f"  Issue: {rec.message}\n")
                f.write(f"  Suggestion: {rec.suggestion}\n\n")
        if verbose:
            print(f"💾 Recommendations saved to: {rec_file}")
    
    return {
        "metrics": metrics,
        "trades": trades,
        "signals": actionable_signals,
        "recommendations": recommendations,
    }


def run_multi_symbol_backtest(
    symbols: List[str],
    strategy_type: str = "trend",
    use_mtf_filter: bool = True,
    mtf_timeframe: str = "1h",
    initial_balance: float = 1000.0,
    use_all_timeframes: bool = True,
    days: Optional[int] = None,  # Ограничить данные последними N днями
) -> Dict[str, Any]:
    """
    Запускает бэктест для нескольких символов одновременно.
    
    Args:
        symbols: Список символов для тестирования (например, ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
        strategy_type: Тип стратегии
        use_mtf_filter: Использовать ли мультитаймфреймовый фильтр
        mtf_timeframe: Таймфрейм для MTF анализа
        initial_balance: Начальный баланс для каждого символа
        use_all_timeframes: Использовать все доступные таймфреймы
    
    Returns:
        Словарь с результатами для каждого символа и сводной статистикой
    """
    results = {}
    all_recommendations = []  # Собираем рекомендации для всех символов
    total_metrics = {
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'total_pnl': 0.0,
        'total_signals': 0,
        'long_signals': 0,
        'short_signals': 0,
    }
    
    print("=" * 80)
    print(f"🚀 MULTI-SYMBOL BACKTEST: {', '.join(symbols)}")
    print(f"   Strategy: {strategy_type.upper()}")
    print(f"   MTF Filter: {'ON' if use_mtf_filter else 'OFF'}")
    if use_mtf_filter:
        print(f"   MTF Mode: {'Multi-timeframe consensus' if use_all_timeframes else f'Single timeframe ({mtf_timeframe})'}")
    print(f"   Data Source: Local CSV files (no API calls)")
    if days:
        print(f"   Period: Last {days} days")
    else:
        print(f"   Period: All available data")
    print("=" * 80)
    
    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"📊 Testing {symbol}")
        print(f"{'='*80}")
        
        # Определяем путь к файлу (пробуем разные форматы)
        symbol_lower = symbol.lower()
        possible_paths = [
            f"data/{symbol_lower}_15m.csv",  # btcusdt_15m.csv
            f"data/{symbol_lower[:3]}_15m.csv",  # btc_15m.csv (fallback)
        ]
        
        csv_path = None
        for path in possible_paths:
            if os.path.exists(path):
                csv_path = path
                break
        
        if csv_path is None:
            print(f"⚠️ CSV file not found for {symbol}. Tried: {', '.join(possible_paths)}")
            print(f"   Skipping {symbol}")
            continue
        
        try:
            result = run_strategy_backtest(
                csv_path=csv_path,
                strategy_type=strategy_type,
                use_mtf_filter=use_mtf_filter,
                mtf_timeframe=mtf_timeframe,
                initial_balance=initial_balance,
                symbol=symbol,
                use_all_timeframes=use_all_timeframes,
                verbose=True,  # Включаем детальное логирование для диагностики
                days=days,  # Передаем ограничение по дням
            )
            
            results[symbol] = result
            metrics = result['metrics']
            
            # Собираем рекомендации
            if 'recommendations' in result and result['recommendations']:
                all_recommendations.extend([(symbol, rec) for rec in result['recommendations']])
            
            # Суммируем метрики
            total_metrics['total_trades'] += metrics.total_trades
            total_metrics['winning_trades'] += metrics.winning_trades
            total_metrics['losing_trades'] += metrics.losing_trades
            total_metrics['total_pnl'] += metrics.total_pnl
            total_metrics['total_signals'] += metrics.total_signals
            total_metrics['long_signals'] += metrics.long_signals
            total_metrics['short_signals'] += metrics.short_signals
            
        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Выводим сводную статистику
    print("\n" + "=" * 80)
    print("📊 SUMMARY - ALL SYMBOLS")
    print("=" * 80)
    
    if total_metrics['total_trades'] > 0:
        overall_win_rate = (total_metrics['winning_trades'] / total_metrics['total_trades']) * 100
    else:
        overall_win_rate = 0.0
    
    print(f"\n📈 Signals (Total):")
    print(f"   Total signals: {total_metrics['total_signals']}")
    print(f"   LONG signals: {total_metrics['long_signals']}")
    print(f"   SHORT signals: {total_metrics['short_signals']}")
    
    print(f"\n💼 Trades (Total):")
    print(f"   Total trades: {total_metrics['total_trades']}")
    print(f"   Winning trades: {total_metrics['winning_trades']}")
    print(f"   Losing trades: {total_metrics['losing_trades']}")
    print(f"   Overall Win Rate: {overall_win_rate:.2f}%")
    
    print(f"\n💰 PnL (Total):")
    print(f"   Total PnL: ${total_metrics['total_pnl']:.2f}")
    print(f"   Average PnL per symbol: ${total_metrics['total_pnl'] / len(results):.2f}")
    
    # Детальная статистика по каждому символу
    print(f"\n📋 Per-Symbol Breakdown:")
    print(f"{'Symbol':<12} {'Trades':<8} {'Win Rate':<10} {'PnL':<12} {'Signals':<8}")
    print("-" * 60)
    
    for symbol, result in results.items():
        metrics = result['metrics']
        win_rate = metrics.win_rate if metrics.total_trades > 0 else 0.0
        print(f"{symbol:<12} {metrics.total_trades:<8} {win_rate:>6.2f}%   ${metrics.total_pnl:>9.2f}  {metrics.total_signals:<8}")
    
    # Сохраняем рекомендации для всех символов
    if all_recommendations:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rec_file = os.path.join("results", f"{strategy_type}_backtest_recommendations_{timestamp}.txt")
        os.makedirs("results", exist_ok=True)
        with open(rec_file, "w", encoding="utf-8") as f:
            f.write(f"{strategy_type.upper()} STRATEGY BACKTEST RECOMMENDATIONS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Strategy: {strategy_type.upper()}\n")
            f.write(f"MTF Filter: {'ON' if use_mtf_filter else 'OFF'}\n")
            f.write(f"Symbols tested: {', '.join(symbols)}\n")
            f.write(f"Total Trades: {total_metrics['total_trades']}\n")
            f.write(f"Overall Win Rate: {overall_win_rate:.2f}%\n")
            f.write(f"Total PnL: ${total_metrics['total_pnl']:.2f}\n\n")
            f.write("RECOMMENDATIONS:\n")
            f.write("=" * 80 + "\n\n")
            for symbol, rec in all_recommendations:
                f.write(f"[{symbol}] [{rec.priority.upper()}] {rec.category.upper()}\n")
                f.write(f"  Issue: {rec.message}\n")
                f.write(f"  Suggestion: {rec.suggestion}\n\n")
        print(f"\n💾 Recommendations saved to: {rec_file}")
    
    return {
        'results': results,
        'summary': total_metrics,
        'overall_win_rate': overall_win_rate,
        'recommendations': all_recommendations,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Backtest strategy with MTF analysis")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV file (for single symbol test)")
    parser.add_argument("--strategy", type=str, choices=["trend", "flat", "momentum"], default="trend", help="Strategy type")
    parser.add_argument("--mtf", action="store_true", help="Use multi-timeframe filter")
    parser.add_argument("--mtf-tf", type=str, choices=["1h", "4h"], default="1h", help="MTF timeframe (if not using all timeframes)")
    parser.add_argument("--no-all-tf", action="store_true", help="Don't use all timeframes, only resample from 15m")
    parser.add_argument("--balance", type=float, default=1000.0, help="Initial balance")
    parser.add_argument("--symbol", type=str, default=None, help="Trading symbol (for single symbol test)")
    parser.add_argument("--multi", action="store_true", help="Test multiple symbols (BTCUSDT, ETHUSDT, SOLUSDT)")
    parser.add_argument("--symbols", type=str, nargs="+", help="List of symbols to test (e.g., BTCUSDT ETHUSDT SOLUSDT)")
    parser.add_argument("--days", type=int, default=30, help="Limit backtest to last N days (1-30, default: 30)")
    
    args = parser.parse_args()
    
    # Валидация days
    if args.days < 1 or args.days > 30:
        print(f"❌ Error: --days must be between 1 and 30, got {args.days}")
        sys.exit(1)
    
    # Если указан флаг --multi или список символов, запускаем мультисимвольный тест
    if args.multi or args.symbols:
        if args.symbols:
            symbols = args.symbols
        else:
            # По умолчанию тестируем 3 основных символа
            symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        
        run_multi_symbol_backtest(
            symbols=symbols,
            strategy_type=args.strategy,
            use_mtf_filter=args.mtf,
            mtf_timeframe=args.mtf_tf,
            initial_balance=args.balance,
            use_all_timeframes=not args.no_all_tf,
            days=args.days,
        )
    else:
        # Одиночный тест (старая логика)
        csv_path = args.csv or "data/btcusdt_15m.csv"
        symbol = args.symbol or "BTCUSDT"
        
        # Проверяем альтернативные пути к файлу (поддержка разных форматов имен)
        if not os.path.exists(csv_path):
            symbol_lower = symbol.lower()
            alternatives = [
                f"data/{symbol_lower[:3]}_15m.csv",  # btc_15m.csv
                f"data/{symbol_lower}_15m.csv",  # btcusdt_15m.csv
            ]
            for alt in alternatives:
                if os.path.exists(alt):
                    csv_path = alt
                    print(f"⚠️ Using alternative CSV: {csv_path}")
                    break
        
        if not os.path.exists(csv_path):
            print(f"❌ CSV file not found: {csv_path}")
            print(f"   Tried paths:")
            symbol_lower = symbol.lower()
            print(f"   - data/{symbol_lower}_15m.csv")
            print(f"   - data/{symbol_lower[:3]}_15m.csv")
            print("\n   Available CSV files in data/:")
            if os.path.exists("data"):
                csv_files = [f for f in os.listdir("data") if f.endswith(".csv") and not f.startswith("backtest_")]
                if csv_files:
                    for f in sorted(csv_files)[:20]:  # Показываем первые 20
                        print(f"   - data/{f}")
                    if len(csv_files) > 20:
                        print(f"   ... and {len(csv_files) - 20} more files")
                else:
                    print("   (no CSV files found)")
            sys.exit(1)
        
        result = run_strategy_backtest(
            csv_path=csv_path,
            strategy_type=args.strategy,
            use_mtf_filter=args.mtf,
            mtf_timeframe=args.mtf_tf,
            initial_balance=args.balance,
            symbol=symbol,
            use_all_timeframes=not args.no_all_tf,
            days=args.days,
        )
