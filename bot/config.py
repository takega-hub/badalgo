from dataclasses import dataclass, field
from typing import Optional, List
import os

from dotenv import load_dotenv


@dataclass
class ApiSettings:
    api_key: str = ""
    api_secret: str = ""
    # По умолчанию используем продакшн Bybit. Для тестнета установи BYBIT_BASE_URL=https://api-testnet.bybit.com
    base_url: str = "https://api.bybit.com"
    
    def __post_init__(self):
        # Загружаем значения из окружения при создании объекта
        if not self.api_key:
            self.api_key = os.getenv("BYBIT_API_KEY", "").strip()
        if not self.api_secret:
            self.api_secret = os.getenv("BYBIT_API_SECRET", "").strip()
        if not self.base_url or self.base_url == "https://api.bybit.com":
            self.base_url = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com").strip()


@dataclass
class StrategyParams:
    # 4H trend filter (ADX)
    adx_threshold: float = 25.0
    adx_length: int = 14
    # 1H direction (PlusDI/MinusDI)
    di_length: int = 14

    # 15m entries
    breakout_lookback: int = 20
    breakout_volume_mult: float = 1.0  # require volume > vol_sma * mult

    # Scaling - averaging on pullback
    sma_length: int = 20
    pullback_tolerance: float = 0.002  # 0.2% band around SMA20
    volume_spike_mult: float = 1.0  # require max(curr, prev) > vol_sma * mult

    # Scaling - pyramiding on consolidation breakout
    consolidation_bars: int = 8
    consolidation_range_pct: float = 0.004  # 0.4% of avg price
    rsi_length: int = 14
    rsi_floor: float = 35.0
    rsi_ceiling: float = 65.0

    # Range/flat strategy parameters (Mean Reversion + Volume Filter)
    bb_length: int = 20  # период для Bollinger Bands
    bb_std: float = 2.0  # стандартное отклонение для Bollinger Bands
    range_rsi_oversold: float = 30.0  # RSI перепроданность (сигнал на покупку)
    range_rsi_overbought: float = 70.0  # RSI перекупленность (сигнал на продажу)
    range_volume_mult: float = 1.3  # максимальный объем для входа (Volume < Volume_SMA * mult)
    range_tp_aggressive: bool = False  # использовать агрессивный TP (до противоположной границы BB)
    range_stop_loss_pct: float = 0.015  # стоп-лосс 1.5% от входа
    range_bb_touch_tolerance_pct: float = 0.002  # допуск для касания BB (0.2%)
    
    # Momentum Breakout strategy parameters (Импульсный пробой для тренда)
    ema_fast_length: int = 20  # быстрая EMA
    ema_slow_length: int = 50  # медленная EMA
    momentum_adx_threshold: float = 25.0  # ADX порог для подтверждения тренда
    momentum_volume_spike_min: float = 1.5  # минимальный объем для входа (Volume > Volume_SMA * mult)
    momentum_volume_spike_max: float = 2.0  # максимальный объем для входа (Volume < Volume_SMA * mult)
    momentum_trailing_stop_ema: bool = True  # использовать EMA 50 как trailing stop
    momentum_ema_timeframe: str = "1h"  # Таймфрейм для EMA (1h или 4h)
    
    
    # Liquidity Sweep strategy parameters (Снятие ликвидности)
    liquidity_donchian_length: int = 20  # Период для Donchian Channels
    liquidity_volume_spike_mult: float = 2.5  # Огромный всплеск объема (вынос стопов) > Volume_SMA * mult
    liquidity_shadow_ratio: float = 0.6  # Минимальная доля тени от общей длины свечи (60%)
    liquidity_reversal_confirmation: bool = True  # Требовать закрытие внутри канала после пробоя
    
    # Smart Money Concepts (SMC) strategy parameters
    smc_fvg_min_gap_pct: float = 0.001  # Минимальный размер FVG в процентах (0.1%)
    smc_fvg_use_atr_filter: bool = True  # Использовать фильтр по ATR для поиска только импульсных движений
    smc_fvg_atr_multiplier: float = 1.5  # Множитель ATR для фильтрации (тело свечи должно быть больше ATR * multiplier)
    smc_ob_lookback: int = 5  # Период для поиска локальных экстремумов при поиске Order Blocks (для 15m оптимально 3-5)
    smc_ob_min_move_pct: float = 0.005  # Минимальное движение после OB в процентах (0.5%)
    smc_ob_require_fvg: bool = True  # Требовать подтверждение FVG для Order Block (Full Scale Displacement)
    smc_touch_tolerance_pct: float = 0.001  # Допустимое отклонение для касания зоны (0.1%)
    smc_max_fvg_age_bars: int = 200  # Максимальный возраст FVG в свечах (после этого зона считается неактивной)
    smc_max_ob_age_bars: int = 300  # Максимальный возраст Order Block в свечах
    smc_rr_ratio: float = 2.5  # Оптимальное RR для M15
    smc_touch_tolerance_pct: float = 0.0001  # Погрешность касания
    smc_enable_session_filter: bool = True  # Включить фильтр торговых сессий
    smc_session_london_start: int = 7
    smc_session_london_end: int = 10
    smc_session_ny_start: int = 12
    smc_session_ny_end: int = 15
    
    # ICT Silver Bullet strategy parameters
    ict_enable_london_session: bool = True  # Торговать в Лондонскую сессию (08:00-16:00 UTC)
    ict_enable_ny_session: bool = True  # Торговать в Нью-Йоркскую сессию (13:00-21:00 UTC)
    ict_alligator_jaw_period: int = 13  # Период челюсти Williams Alligator
    ict_alligator_teeth_period: int = 8  # Период зубов Williams Alligator
    ict_alligator_lips_period: int = 5  # Период губ Williams Alligator
    ict_alligator_jaw_shift: int = 8  # Сдвиг челюсти
    ict_alligator_teeth_shift: int = 5  # Сдвиг зубов
    ict_alligator_lips_shift: int = 3  # Сдвиг губ
    ict_fvg_max_age_bars: int = 20  # Максимальный возраст FVG для входа (в свечах)
    ict_liquidity_lookback_days: int = 1  # Количество дней для поиска ликвидности
    ict_atr_multiplier_sl: float = 2.0  # Множитель ATR для стоп-лосса
    ict_rr_ratio: float = 2.0  # Минимальное соотношение Risk/Reward (1:2)


@dataclass
class RiskParams:
    max_position_usd: float = 200.0
    # Защита от повторных убыточных сделок
    enable_loss_cooldown: bool = True  # Включить защиту от повторных убыточных сделок
    loss_cooldown_minutes: int = 60  # Период "охлаждения" в минутах после убыточной сделки
    max_consecutive_losses: int = 2  # Максимальное количество убыточных сделок подряд, после которого блокируем
    base_order_usd: float = 50.0
    add_order_usd: float = 50.0
    stop_loss_pct: float = 0.01  # 1% stop from entry
    take_profit_pct: float = 0.02  # 2% TP
    # Процент от баланса для одной сделки (если > 0, используется вместо base_order_usd)
    balance_percent_per_trade: float = 20.0  # 0 = использовать base_order_usd, > 0 = процент от баланса
    
    # Trailing Stop Loss
    enable_trailing_stop: bool = True  # Включить trailing stop loss
    trailing_stop_activation_pct: float = 0.005  # Активировать trailing stop при прибыли 0.5%
    trailing_stop_distance_pct: float = 0.003  # Расстояние trailing stop от максимума (0.3%)
    
    # Частичное закрытие позиции
    enable_partial_close: bool = True  # Включить частичное закрытие
    partial_close_pct: float = 0.5  # Закрывать 50% позиции
    partial_close_at_tp_pct: float = 0.5  # При достижении 50% пути к TP
    
    # Защита прибыли
    enable_profit_protection: bool = True  # Включить защиту прибыли
    profit_protection_activation_pct: float = 0.01  # Активировать защиту при прибыли 1%
    profit_protection_retreat_pct: float = 0.003  # Закрывать при откате 0.3% от максимума
    
    # Безубыток
    enable_breakeven: bool = True  # Включить перемещение SL в безубыток
    breakeven_activation_pct: float = 0.005  # Активировать безубыток при прибыли 0.5%
    
    # Умное добавление к позиции (averaging/pyramiding)
    enable_smart_add: bool = True  # Включить умное добавление
    smart_add_pullback_pct: float = 0.002  # Минимальный откат для добавления (0.2%)
    max_add_count: int = 2  # Максимальное количество докупок (2 = можно добавить 2 раза)
    smart_add_tp_sl_progress_pct: float = 0.5  # Добавлять когда цена прошла N% пути к TP или SL (50%)
    smart_add_adjust_sl: bool = True  # Пересчитывать SL после докупки по новой средней цене
    
    # Защита от входа на границах (ATR анализ)
    enable_atr_entry_filter: bool = True  # Включить фильтр входа по ATR
    max_atr_progress_pct: float = 0.7  # Максимальный процент ATR, который уже пройден (70% = не входить если прошло >70% ATR)


@dataclass
class AppSettings:
    # Многопарная торговля
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT"])  # Все доступные пары
    active_symbols: List[str] = field(default_factory=lambda: ["BTCUSDT"])  # Активные для торговли
    primary_symbol: str = "BTCUSDT"  # Основная пара для UI (по умолчанию)
    
    # Обратная совместимость (deprecated, используйте primary_symbol)
    symbol: str = "BTCUSDT"  # Доступные пары: BTCUSDT, ETHUSDT, SOLUSDT
    
    timeframe: str = "15m"
    live_poll_seconds: int = 60  # пауза между циклами live-бота
    kline_limit: int = 1000      # сколько свечей тянуть для сигналов (SMC требует много истории)
    strategy: StrategyParams = field(default_factory=StrategyParams)
    risk: RiskParams = field(default_factory=RiskParams)
    api: ApiSettings = field(default_factory=ApiSettings)
    leverage: int = 10
    # ML стратегия
    strategy_type: str = "rule-based"  # "rule-based", "ml", "hybrid"
    ml_model_path: Optional[str] = None  # Путь к обученной ML модели (для основной пары)
    ml_model_type_for_all: Optional[str] = None  # Тип модели для всех пар: "rf", "xgb", "ensemble" (None = авто-выбор)
    ml_confidence_threshold: float = 0.5  # Минимальная уверенность ML модели для открытия позиции (0-1)
    ml_min_signal_strength: str = "слабое"  # Минимальная сила сигнала: "слабое", "умеренное", "среднее", "сильное", "очень_сильное"
    ml_stability_filter: bool = True  # Фильтр стабильности: игнорировать слабые сигналы при смене направления
    ml_target_profit_pct_margin: float = 25.0  # Целевая прибыль от маржи в % (20-30%)
    ml_max_loss_pct_margin: float = 10.0  # Максимальный убыток от маржи в % (обычно 50% от прибыли)
    # Выбор активных стратегий (можно использовать несколько одновременно)
    enable_trend_strategy: bool = True  # Трендовая стратегия (старая)
    enable_flat_strategy: bool = True   # Флэтовая стратегия (старая)
    enable_ml_strategy: bool = False    # ML стратегия
    enable_momentum_strategy: bool = False  # Стратегия "Импульсный пробой" (новая для тренда)
    enable_liquidity_sweep_strategy: bool = False  # Стратегия "Liquidity Sweep" (снятие ликвидности)
    enable_smc_strategy: bool = False  # Smart Money Concepts стратегия
    enable_ict_strategy: bool = False  # ICT Silver Bullet стратегия
    # Приоритетная стратегия при конфликте сигналов
    strategy_priority: str = "trend"  # "trend", "flat", "ml", "momentum", "liquidity", "smc", "ict", "hybrid", "confluence"
    
    def __post_init__(self):
        """Инициализация после создания dataclass"""
        # Синхронизируем symbol с primary_symbol для обратной совместимости
        if self.symbol == "BTCUSDT" and self.primary_symbol != "BTCUSDT":
            self.symbol = self.primary_symbol
        elif self.primary_symbol == "BTCUSDT" and self.symbol != "BTCUSDT":
            self.primary_symbol = self.symbol
        
        # Валидация символов
        available_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        if not self.symbols:
            self.symbols = available_symbols
        
        # Проверяем, что все активные символы в списке доступных
        self.active_symbols = [s for s in self.active_symbols if s in available_symbols]
        if not self.active_symbols:
            self.active_symbols = [self.primary_symbol] if self.primary_symbol in available_symbols else [available_symbols[0]]
        
        # Проверяем primary_symbol
        if self.primary_symbol not in available_symbols:
            self.primary_symbol = self.active_symbols[0] if self.active_symbols else available_symbols[0]


def load_settings() -> AppSettings:
    """
    Returns AppSettings with env overrides applied to API fields.
    Extend this if you want to pull strategy/risk params from env or files.
    """
    # Подтягиваем переменные из .env, если файл есть
    import pathlib
    project_root = pathlib.Path(__file__).parent.parent
    env_path = project_root / ".env"
    
    # Загружаем .env файл
    load_dotenv(dotenv_path=env_path, override=True)
    
    # Если os.getenv не работает, пробуем через dotenv_values (для обработки BOM)
    from dotenv import dotenv_values
    env_values = dotenv_values(dotenv_path=env_path)
    
    # Очищаем ключи от BOM (Byte Order Mark)
    cleaned_env_values = {}
    for key, value in env_values.items():
        cleaned_key = key.lstrip('\ufeff').strip()
        cleaned_env_values[cleaned_key] = value.strip() if value else ""
    
    # Получаем значения из окружения или из dotenv_values
    api_key_from_env = os.getenv("BYBIT_API_KEY", "").strip()
    if not api_key_from_env:
        api_key_from_env = (
            cleaned_env_values.get("BYBIT_API_KEY", "") or
            env_values.get("\ufeffBYBIT_API_KEY", "") or
            env_values.get("BYBIT_API_KEY", "")
        ).strip()
    
    api_secret_from_env = os.getenv("BYBIT_API_SECRET", "").strip()
    if not api_secret_from_env:
        api_secret_from_env = cleaned_env_values.get("BYBIT_API_SECRET", "").strip()
    
    # Очищаем от невидимых символов
    api_key_from_env = "".join(c for c in api_key_from_env if c.isprintable() or c.isspace()).strip()
    api_secret_from_env = "".join(c for c in api_secret_from_env if c.isprintable() or c.isspace()).strip()
    
    # Создаем настройки
    settings = AppSettings()
    settings.api.api_key = api_key_from_env
    settings.api.api_secret = api_secret_from_env
    settings.api.base_url = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com").strip()
    
    # Загружаем настройки стратегий из .env
    # Используем также cleaned_env_values для надежности
    enable_trend_raw = os.getenv("ENABLE_TREND_STRATEGY", "")
    if not enable_trend_raw:
        enable_trend_raw = cleaned_env_values.get("ENABLE_TREND_STRATEGY", "")
    enable_trend = enable_trend_raw.strip().lower() if enable_trend_raw else ""
    
    enable_flat_raw = os.getenv("ENABLE_FLAT_STRATEGY", "")
    if not enable_flat_raw:
        enable_flat_raw = cleaned_env_values.get("ENABLE_FLAT_STRATEGY", "")
    enable_flat = enable_flat_raw.strip().lower() if enable_flat_raw else ""
    
    enable_ml_raw = os.getenv("ENABLE_ML_STRATEGY", "")
    if not enable_ml_raw:
        enable_ml_raw = cleaned_env_values.get("ENABLE_ML_STRATEGY", "")
    enable_ml = enable_ml_raw.strip().lower() if enable_ml_raw else ""
    
    # Загружаем настройки новых стратегий
    enable_momentum_raw = os.getenv("ENABLE_MOMENTUM_STRATEGY", "")
    if not enable_momentum_raw:
        enable_momentum_raw = cleaned_env_values.get("ENABLE_MOMENTUM_STRATEGY", "")
    enable_momentum = enable_momentum_raw.strip().lower() if enable_momentum_raw else ""
    
    
    enable_liquidity_raw = os.getenv("ENABLE_LIQUIDITY_SWEEP_STRATEGY", "")
    if not enable_liquidity_raw:
        enable_liquidity_raw = cleaned_env_values.get("ENABLE_LIQUIDITY_SWEEP_STRATEGY", "")
    enable_liquidity = enable_liquidity_raw.strip().lower() if enable_liquidity_raw else ""
    
    enable_smc_raw = os.getenv("ENABLE_SMC_STRATEGY", "")
    if not enable_smc_raw:
        enable_smc_raw = cleaned_env_values.get("ENABLE_SMC_STRATEGY", "")
    enable_smc = enable_smc_raw.strip().lower() if enable_smc_raw else ""
    
    enable_ict_raw = os.getenv("ENABLE_ICT_STRATEGY", "")
    if not enable_ict_raw:
        enable_ict_raw = cleaned_env_values.get("ENABLE_ICT_STRATEGY", "")
    enable_ict = enable_ict_raw.strip().lower() if enable_ict_raw else ""
    
    smc_max_fvg_age_bars = os.getenv("SMC_MAX_FVG_AGE_BARS", "").strip()
    if smc_max_fvg_age_bars:
        try:
            settings.strategy.smc_max_fvg_age_bars = int(smc_max_fvg_age_bars)
        except ValueError:
            pass
            
    smc_max_ob_age_bars = os.getenv("SMC_MAX_OB_AGE_BARS", "").strip()
    if smc_max_ob_age_bars:
        try:
            settings.strategy.smc_max_ob_age_bars = int(smc_max_ob_age_bars)
        except ValueError:
            pass

    smc_rr_ratio = os.getenv("SMC_RR_RATIO", "").strip()
    if smc_rr_ratio:
        try:
            settings.strategy.smc_rr_ratio = float(smc_rr_ratio)
        except ValueError:
            pass

    enable_smc_session = os.getenv("SMC_ENABLE_SESSION_FILTER", "").strip().lower()
    if enable_smc_session:
        settings.strategy.smc_enable_session_filter = enable_smc_session in ("true", "1", "yes")
            
    # Пробуем получить TRADING_SYMBOL из разных источников
    trading_symbol_raw = os.getenv("TRADING_SYMBOL", "")
    if not trading_symbol_raw:
        trading_symbol_raw = cleaned_env_values.get("TRADING_SYMBOL", "")
    
    # Если все еще не найдено, читаем напрямую из файла (последнее значение)
    if not trading_symbol_raw and env_path.exists():
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Ищем последнее вхождение TRADING_SYMBOL=
                lines = content.split('\n')
                for line in reversed(lines):  # Идем с конца файла
                    line_stripped = line.strip()
                    if line_stripped.startswith('TRADING_SYMBOL='):
                        trading_symbol_raw = line_stripped.split('=', 1)[1].strip()
                        print(f"[config] Found TRADING_SYMBOL by reading .env file directly: '{trading_symbol_raw}'")
                        break
        except Exception as e:
            print(f"[config] Error reading .env file directly: {e}")
    
    trading_symbol = trading_symbol_raw.strip() if trading_symbol_raw else ""
    
    print(f"[config] TRADING_SYMBOL from os.getenv: '{os.getenv('TRADING_SYMBOL', 'NOT_FOUND')}'")
    print(f"[config] TRADING_SYMBOL from cleaned_env_values: '{cleaned_env_values.get('TRADING_SYMBOL', 'NOT_FOUND')}'")
    print(f"[config] Final trading_symbol: '{trading_symbol}'")
    
    # Загружаем активные символы из .env (многопарная торговля)
    available_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    active_symbols_raw = os.getenv("ACTIVE_SYMBOLS", "")
    if not active_symbols_raw:
        active_symbols_raw = cleaned_env_values.get("ACTIVE_SYMBOLS", "")
    
    if active_symbols_raw:
        # Парсим список символов (comma-separated)
        active_symbols_list = [s.strip().upper() for s in active_symbols_raw.split(",") if s.strip()]
        # Фильтруем только валидные символы
        active_symbols_list = [s for s in active_symbols_list if s in available_symbols]
        if active_symbols_list:
            settings.active_symbols = active_symbols_list
            print(f"[config] ACTIVE_SYMBOLS loaded from .env: {settings.active_symbols}")
    elif trading_symbol and trading_symbol in available_symbols:
        # Обратная совместимость: если ACTIVE_SYMBOLS не задан, используем TRADING_SYMBOL
        settings.active_symbols = [trading_symbol]
        print(f"[config] ACTIVE_SYMBOLS not found, using TRADING_SYMBOL for backward compatibility: {settings.active_symbols}")
    else:
        print(f"[config] No ACTIVE_SYMBOLS or TRADING_SYMBOL, using default: {settings.active_symbols}")
    
    # Загружаем основной символ для UI
    primary_symbol_raw = os.getenv("PRIMARY_SYMBOL", "")
    if not primary_symbol_raw:
        primary_symbol_raw = cleaned_env_values.get("PRIMARY_SYMBOL", "")
    
    if primary_symbol_raw and primary_symbol_raw.strip().upper() in available_symbols:
        settings.primary_symbol = primary_symbol_raw.strip().upper()
        print(f"[config] PRIMARY_SYMBOL loaded from .env: {settings.primary_symbol}")
    elif settings.active_symbols:
        # Если PRIMARY_SYMBOL не задан, используем первый активный
        settings.primary_symbol = settings.active_symbols[0]
        print(f"[config] PRIMARY_SYMBOL not found, using first active symbol: {settings.primary_symbol}")
    elif trading_symbol and trading_symbol in available_symbols:
        # Обратная совместимость: используем TRADING_SYMBOL
        settings.primary_symbol = trading_symbol
        print(f"[config] PRIMARY_SYMBOL not found, using TRADING_SYMBOL for backward compatibility: {settings.primary_symbol}")
    
    # Синхронизируем symbol с primary_symbol для обратной совместимости
    settings.symbol = settings.primary_symbol
    
    # Применяем настройки стратегий (если заданы в .env, переопределяем значения по умолчанию)
    # Проверяем, что переменная присутствует в .env (используем cleaned_env_values для надежности)
    if "ENABLE_TREND_STRATEGY" in cleaned_env_values or enable_trend_raw:
        settings.enable_trend_strategy = enable_trend in ("true", "1", "yes")
    if "ENABLE_FLAT_STRATEGY" in cleaned_env_values or enable_flat_raw:
        settings.enable_flat_strategy = enable_flat in ("true", "1", "yes")
    if "ENABLE_ML_STRATEGY" in cleaned_env_values or enable_ml_raw:
        settings.enable_ml_strategy = enable_ml in ("true", "1", "yes")
    if "ENABLE_MOMENTUM_STRATEGY" in cleaned_env_values or enable_momentum_raw:
        settings.enable_momentum_strategy = enable_momentum in ("true", "1", "yes")
    if "ENABLE_LIQUIDITY_SWEEP_STRATEGY" in cleaned_env_values or enable_liquidity_raw:
        settings.enable_liquidity_sweep_strategy = enable_liquidity in ("true", "1", "yes")
    
    if "ENABLE_SMC_STRATEGY" in cleaned_env_values or enable_smc_raw:
        settings.enable_smc_strategy = enable_smc in ("true", "1", "yes")
    
    if "ENABLE_ICT_STRATEGY" in cleaned_env_values or enable_ict_raw:
        settings.enable_ict_strategy = enable_ict in ("true", "1", "yes")
    
    # Логируем загруженные настройки для отладки
    print(f"[config] Loaded strategy settings from .env:")
    print(f"  TRADING_SYMBOL='{trading_symbol_raw}' -> {settings.symbol}")
    print(f"  ENABLE_TREND_STRATEGY='{enable_trend_raw}' -> {settings.enable_trend_strategy}")
    print(f"  ENABLE_FLAT_STRATEGY='{enable_flat_raw}' -> {settings.enable_flat_strategy}")
    print(f"  ENABLE_ML_STRATEGY='{enable_ml_raw}' -> {settings.enable_ml_strategy}")
    print(f"  ENABLE_MOMENTUM_STRATEGY='{enable_momentum_raw}' -> {settings.enable_momentum_strategy}")
    print(f"  ENABLE_LIQUIDITY_SWEEP_STRATEGY='{enable_liquidity_raw}' -> {settings.enable_liquidity_sweep_strategy}")
    print(f"  ENABLE_SMC_STRATEGY='{enable_smc_raw}' -> {settings.enable_smc_strategy}")
    print(f"  ENABLE_ICT_STRATEGY='{enable_ict_raw}' -> {settings.enable_ict_strategy}")
    
    # Загружаем приоритет стратегии
    strategy_priority_raw = os.getenv("STRATEGY_PRIORITY", "")
    if not strategy_priority_raw:
        strategy_priority_raw = cleaned_env_values.get("STRATEGY_PRIORITY", "")
    strategy_priority = strategy_priority_raw.strip().lower() if strategy_priority_raw else ""
    allowed_priorities = ("trend", "flat", "ml", "momentum", "liquidity", "smc", "hybrid", "confluence")
    if strategy_priority in allowed_priorities:
        settings.strategy_priority = strategy_priority
        print(f"[config] STRATEGY_PRIORITY loaded from .env: {settings.strategy_priority}")
    else:
        print(f"[config] No STRATEGY_PRIORITY in .env or invalid value '{strategy_priority}', using default: {settings.strategy_priority}")
    
    # Автоматически ищем модель для выбранного символа
    # Проверяем, соответствует ли текущая модель выбранному символу
    if settings.symbol:
        import pathlib
        models_dir = pathlib.Path(__file__).parent.parent / "ml_models"
        if models_dir.exists():
            # Проверяем, соответствует ли текущая модель символу
            current_model_matches = False
            if settings.ml_model_path:
                model_path_obj = pathlib.Path(settings.ml_model_path)
                if model_path_obj.exists() and f"_{settings.symbol}_" in model_path_obj.name:
                    current_model_matches = True
                    print(f"[config] Current ML model matches symbol {settings.symbol}: {settings.ml_model_path}")
            
            # Если модель не соответствует символу или не задана, ищем новую
            if not current_model_matches:
                found_model = None
                model_type_preference = getattr(settings, 'ml_model_type_for_all', None)
                
                if model_type_preference:
                    # Ищем модель указанного типа
                    pattern = f"{model_type_preference}_{settings.symbol}_*.pkl"
                    for model_file in sorted(models_dir.glob(pattern), reverse=True):
                        if model_file.is_file():
                            found_model = str(model_file)
                            break
                else:
                    # Автоматический выбор: предпочитаем ensemble > rf > xgb
                    for model_type in ["ensemble", "rf", "xgb"]:
                        pattern = f"{model_type}_{settings.symbol}_*.pkl"
                        for model_file in sorted(models_dir.glob(pattern), reverse=True):
                            if model_file.is_file():
                                found_model = str(model_file)
                                break
                        if found_model:
                            break
                
                if found_model:
                    old_model = settings.ml_model_path
                    settings.ml_model_path = found_model
                    if old_model:
                        print(f"[config] ML model changed for {settings.symbol}: {old_model} -> {found_model}")
                    else:
                        print(f"[config] Auto-found ML model for {settings.symbol}: {found_model}")
                else:
                    if settings.ml_model_path:
                        print(f"[config] No ML model found for {settings.symbol}, clearing model path (was: {settings.ml_model_path})")
                    settings.ml_model_path = None
    
    # Настройки ML-стратегии (можно переопределить через переменные окружения)
    # ВАЖНО: Сначала загружаем настройки из .env, потом проверяем наличие модели
    ml_strategy_type = os.getenv("ML_STRATEGY_TYPE", "").strip()
    ml_model_path = os.getenv("ML_MODEL_PATH", "").strip()
    ml_model_type_for_all = os.getenv("ML_MODEL_TYPE_FOR_ALL", "").strip().lower() or None
    ml_confidence = os.getenv("ML_CONFIDENCE_THRESHOLD", "").strip()
    ml_min_strength = os.getenv("ML_MIN_SIGNAL_STRENGTH", "").strip()
    ml_stability = os.getenv("ML_STABILITY_FILTER", "").strip()
    
    # Применяем настройки из .env ПЕРЕД автоматическим поиском модели
    if ml_strategy_type:
        settings.strategy_type = ml_strategy_type
    if ml_model_path:
        settings.ml_model_path = ml_model_path
    if ml_model_type_for_all and ml_model_type_for_all.lower() in ("rf", "xgb", "ensemble"):
        settings.ml_model_type_for_all = ml_model_type_for_all.lower()
        print(f"[config] ML_MODEL_TYPE_FOR_ALL loaded from .env: {settings.ml_model_type_for_all}")
    if ml_confidence:
        try:
            settings.ml_confidence_threshold = float(ml_confidence)
            print(f"[config] ML_CONFIDENCE_THRESHOLD loaded from .env: {settings.ml_confidence_threshold}")
        except ValueError:
            print(f"[config] Warning: Invalid ML_CONFIDENCE_THRESHOLD value: {ml_confidence}")
    
    if ml_min_strength:
        valid_strengths = ["слабое", "умеренное", "среднее", "сильное", "очень_сильное"]
        if ml_min_strength.lower() in valid_strengths:
            settings.ml_min_signal_strength = ml_min_strength.lower()
            print(f"[config] ML_MIN_SIGNAL_STRENGTH loaded from .env: {settings.ml_min_signal_strength}")
    
    if ml_stability:
        settings.ml_stability_filter = ml_stability.lower() in ("true", "1", "yes", "on")
        print(f"[config] ML_STABILITY_FILTER loaded from .env: {settings.ml_stability_filter}")
    
    # Если модель не задана через env и не найдена автоматически выше, проверяем наличие модели и используем её
    # НО НЕ перезаписываем настройки, если они уже заданы в .env или найдены автоматически
    if not ml_strategy_type and not ml_model_path and not settings.ml_model_path:
        models_dir = pathlib.Path(__file__).parent.parent / "ml_models"
        if models_dir.exists():
            found_model = None
            
            # Используем ml_model_type_for_all, если задан
            model_type_preference = getattr(settings, 'ml_model_type_for_all', None)
            
            if settings.symbol:
                # Ищем модель для текущего символа
                if model_type_preference:
                    # Ищем модель указанного типа
                    pattern = f"{model_type_preference}_{settings.symbol}_*.pkl"
                    for model_file in sorted(models_dir.glob(pattern), reverse=True):
                        if model_file.is_file():
                            found_model = str(model_file)
                            break
                else:
                    # Автоматический выбор: предпочитаем ensemble > rf > xgb
                    for model_type in ["ensemble", "rf", "xgb"]:
                        pattern = f"{model_type}_{settings.symbol}_*.pkl"
                        for model_file in sorted(models_dir.glob(pattern), reverse=True):
                            if model_file.is_file():
                                found_model = str(model_file)
                                break
                        if found_model:
                            break
                
                if found_model:
                    print(f"[config] Found ML model for {settings.symbol}, enabling ML strategy: {found_model}")
                    settings.strategy_type = "ml"
                    settings.ml_model_path = found_model
                    print(f"[config] ML Strategy enabled: model={settings.ml_model_path}, confidence={settings.ml_confidence_threshold}")
            else:
                # Если символ не задан, пробуем найти любую модель (fallback на ETHUSDT)
                if model_type_preference:
                    pattern = f"{model_type_preference}_ETHUSDT_*.pkl"
                    for model_file in sorted(models_dir.glob(pattern), reverse=True):
                        if model_file.is_file():
                            found_model = str(model_file)
                            break
                else:
                    # Fallback: ищем rf_ETHUSDT_15.pkl
                    model_path = models_dir / "rf_ETHUSDT_15.pkl"
                    if model_path.exists():
                        found_model = str(model_path)
                
                if found_model:
                    print(f"[config] Found ML model (fallback), enabling ML strategy: {found_model}")
                    settings.strategy_type = "ml"
                    settings.ml_model_path = found_model
                    settings.symbol = "ETHUSDT"  # ML модель обучена на ETH
                    print(f"[config] ML Strategy enabled: model={settings.ml_model_path}, confidence={settings.ml_confidence_threshold}")
    
    # Загружаем параметры TP/SL для ML стратегии
    ml_target_profit = os.getenv("ML_TARGET_PROFIT_PCT_MARGIN", "").strip()
    if ml_target_profit:
        try:
            settings.ml_target_profit_pct_margin = float(ml_target_profit)
        except ValueError:
            pass
    
    ml_max_loss = os.getenv("ML_MAX_LOSS_PCT_MARGIN", "").strip()
    if ml_max_loss:
        try:
            settings.ml_max_loss_pct_margin = float(ml_max_loss)
        except ValueError:
            pass
    
    # Загружаем параметры стратегии (Trend/Flat) из .env
    adx_threshold = os.getenv("ADX_THRESHOLD", "").strip()
    if adx_threshold:
        try:
            settings.strategy.adx_threshold = float(adx_threshold)
        except ValueError:
            pass
    
    adx_length = os.getenv("ADX_LENGTH", "").strip()
    if adx_length:
        try:
            settings.strategy.adx_length = int(adx_length)
        except ValueError:
            pass
    
    di_length = os.getenv("DI_LENGTH", "").strip()
    if di_length:
        try:
            settings.strategy.di_length = int(di_length)
        except ValueError:
            pass
    
    breakout_lookback = os.getenv("BREAKOUT_LOOKBACK", "").strip()
    if breakout_lookback:
        try:
            settings.strategy.breakout_lookback = int(breakout_lookback)
        except ValueError:
            pass
    
    breakout_volume_mult = os.getenv("BREAKOUT_VOLUME_MULT", "").strip()
    if breakout_volume_mult:
        try:
            settings.strategy.breakout_volume_mult = float(breakout_volume_mult)
        except ValueError:
            pass
    
    # Momentum strategy parameters
    momentum_adx_threshold = os.getenv("MOMENTUM_ADX_THRESHOLD", "").strip()
    if momentum_adx_threshold:
        try:
            settings.strategy.momentum_adx_threshold = float(momentum_adx_threshold)
        except ValueError:
            pass
    
    sma_length = os.getenv("SMA_LENGTH", "").strip()
    if sma_length:
        try:
            settings.strategy.sma_length = int(sma_length)
        except ValueError:
            pass
    
    pullback_tolerance = os.getenv("PULLBACK_TOLERANCE", "").strip()
    if pullback_tolerance:
        try:
            settings.strategy.pullback_tolerance = float(pullback_tolerance)
        except ValueError:
            pass
    
    volume_spike_mult = os.getenv("VOLUME_SPIKE_MULT", "").strip()
    if volume_spike_mult:
        try:
            settings.strategy.volume_spike_mult = float(volume_spike_mult)
        except ValueError:
            pass
    
    consolidation_bars = os.getenv("CONSOLIDATION_BARS", "").strip()
    if consolidation_bars:
        try:
            settings.strategy.consolidation_bars = int(consolidation_bars)
        except ValueError:
            pass
    
    consolidation_range_pct = os.getenv("CONSOLIDATION_RANGE_PCT", "").strip()
    if consolidation_range_pct:
        try:
            settings.strategy.consolidation_range_pct = float(consolidation_range_pct)
        except ValueError:
            pass
    
    rsi_length = os.getenv("RSI_LENGTH", "").strip()
    if rsi_length:
        try:
            settings.strategy.rsi_length = int(rsi_length)
        except ValueError:
            pass
    
    rsi_floor = os.getenv("RSI_FLOOR", "").strip()
    if rsi_floor:
        try:
            settings.strategy.rsi_floor = float(rsi_floor)
        except ValueError:
            pass
    
    rsi_ceiling = os.getenv("RSI_CEILING", "").strip()
    if rsi_ceiling:
        try:
            settings.strategy.rsi_ceiling = float(rsi_ceiling)
        except ValueError:
            pass
    
    # Загружаем параметры Range стратегии из .env
    bb_length = os.getenv("BB_LENGTH", "").strip()
    if bb_length:
        try:
            settings.strategy.bb_length = int(bb_length)
        except ValueError:
            pass
    
    bb_std = os.getenv("BB_STD", "").strip()
    if bb_std:
        try:
            settings.strategy.bb_std = float(bb_std)
        except ValueError:
            pass
    
    range_rsi_oversold = os.getenv("RANGE_RSI_OVERSOLD", "").strip()
    if range_rsi_oversold:
        try:
            settings.strategy.range_rsi_oversold = float(range_rsi_oversold)
        except ValueError:
            pass
    
    range_rsi_overbought = os.getenv("RANGE_RSI_OVERBOUGHT", "").strip()
    if range_rsi_overbought:
        try:
            settings.strategy.range_rsi_overbought = float(range_rsi_overbought)
        except ValueError:
            pass
    
    range_volume_mult = os.getenv("RANGE_VOLUME_MULT", "").strip()
    if range_volume_mult:
        try:
            settings.strategy.range_volume_mult = float(range_volume_mult)
        except ValueError:
            pass
    
    range_tp_aggressive = os.getenv("RANGE_TP_AGGRESSIVE", "").strip()
    if range_tp_aggressive:
        settings.strategy.range_tp_aggressive = range_tp_aggressive.lower() in ("true", "1", "yes", "on")
    
    range_stop_loss_pct = os.getenv("RANGE_STOP_LOSS_PCT", "").strip()
    if range_stop_loss_pct:
        try:
            rsl_value = float(range_stop_loss_pct)
            # Преобразуем проценты в доли
            # Если значение >= 1, делим на 100 (например, 1.5 -> 0.015 = 1.5%)
            # Если значение < 1, считаем что это уже доли (например, 0.015 = 1.5%)
            if rsl_value >= 1.0:
                rsl_value = rsl_value / 100.0  # Преобразуем проценты в доли (1.5 -> 0.015 = 1.5%)
                print(f"[config] ⚠️ RANGE_STOP_LOSS_PCT={range_stop_loss_pct} interpreted as percentage, converted to {rsl_value:.4f} (fraction, divided by 100)")
            # Валидация: Range SL должен быть от 0.1% до 50%
            if rsl_value < 0.001 or rsl_value > 0.5:
                print(f"[config] ⚠️ WARNING: RANGE_STOP_LOSS_PCT={rsl_value:.4f} ({rsl_value*100:.2f}%) is out of reasonable range (0.1%-50%), using default 0.015 (1.5%)")
                rsl_value = 0.015
            settings.strategy.range_stop_loss_pct = rsl_value
        except ValueError:
            pass
    
    range_bb_touch_tolerance_pct = os.getenv("RANGE_BB_TOUCH_TOLERANCE_PCT", "").strip()
    if range_bb_touch_tolerance_pct:
        try:
            settings.strategy.range_bb_touch_tolerance_pct = float(range_bb_touch_tolerance_pct)
        except ValueError:
            pass
    
    # Загружаем параметры управления рисками из .env
    max_position_usd = os.getenv("MAX_POSITION_USD", "").strip()
    if max_position_usd:
        try:
            settings.risk.max_position_usd = float(max_position_usd)
        except ValueError:
            pass
    
    base_order_usd = os.getenv("BASE_ORDER_USD", "").strip()
    if base_order_usd:
        try:
            settings.risk.base_order_usd = float(base_order_usd)
        except ValueError:
            pass
    
    add_order_usd = os.getenv("ADD_ORDER_USD", "").strip()
    if add_order_usd:
        try:
            settings.risk.add_order_usd = float(add_order_usd)
        except ValueError:
            pass
    
    stop_loss_pct = os.getenv("STOP_LOSS_PCT", "").strip()
    if stop_loss_pct:
        try:
            sl_value = float(stop_loss_pct)
            # Преобразуем проценты в доли
            # Если значение >= 1, делим на 100 (например, 7 -> 0.07 = 7%)
            # Если значение < 1, считаем что это уже доли (например, 0.01 = 1%)
            if sl_value >= 1.0:
                sl_value = sl_value / 100.0  # Преобразуем проценты в доли (7 -> 0.07 = 7%)
                print(f"[config] ⚠️ STOP_LOSS_PCT={stop_loss_pct} interpreted as percentage, converted to {sl_value:.4f} (fraction, divided by 100)")
            # Валидация: SL должен быть от 0.1% до 50%
            if sl_value < 0.001 or sl_value > 0.5:
                print(f"[config] ⚠️ WARNING: STOP_LOSS_PCT={sl_value:.4f} ({sl_value*100:.2f}%) is out of reasonable range (0.1%-50%), using default 0.01 (1%)")
                sl_value = 0.01
            settings.risk.stop_loss_pct = sl_value
        except ValueError:
            pass
    
    take_profit_pct = os.getenv("TAKE_PROFIT_PCT", "").strip()
    if take_profit_pct:
        try:
            tp_value = float(take_profit_pct)
            # Преобразуем проценты в доли
            # Если значение >= 1, делим на 100 (например, 21 -> 0.21 = 21%)
            # Если значение < 1, считаем что это уже доли (например, 0.02 = 2%)
            if tp_value >= 1.0:
                tp_value = tp_value / 100.0  # Преобразуем проценты в доли (21 -> 0.21 = 21%)
                print(f"[config] ⚠️ TAKE_PROFIT_PCT={take_profit_pct} interpreted as percentage, converted to {tp_value:.4f} (fraction, divided by 100)")
            # Валидация: TP должен быть от 0.5% до 100%
            if tp_value < 0.005 or tp_value > 1.0:
                print(f"[config] ⚠️ WARNING: TAKE_PROFIT_PCT={tp_value:.4f} ({tp_value*100:.2f}%) is out of reasonable range (0.5%-100%), using default 0.02 (2%)")
                tp_value = 0.02
            settings.risk.take_profit_pct = tp_value
        except ValueError:
            pass
    
    balance_percent_per_trade = os.getenv("BALANCE_PERCENT_PER_TRADE", "").strip()
    if balance_percent_per_trade:
        try:
            settings.risk.balance_percent_per_trade = float(balance_percent_per_trade)
        except ValueError:
            pass
    
    # Загружаем настройки trailing stop из .env
    enable_trailing_stop = os.getenv("ENABLE_TRAILING_STOP", "").strip().lower()
    if enable_trailing_stop:
        settings.risk.enable_trailing_stop = enable_trailing_stop in ("true", "1", "yes", "on")
    
    trailing_stop_activation = os.getenv("TRAILING_STOP_ACTIVATION_PCT", "").strip()
    if trailing_stop_activation:
        try:
            activation_value = float(trailing_stop_activation)
            if activation_value >= 1.0:
                activation_value = activation_value / 100.0
            settings.risk.trailing_stop_activation_pct = activation_value
        except ValueError:
            pass
    
    trailing_stop_distance = os.getenv("TRAILING_STOP_DISTANCE_PCT", "").strip()
    if trailing_stop_distance:
        try:
            distance_value = float(trailing_stop_distance)
            if distance_value >= 1.0:
                distance_value = distance_value / 100.0
            settings.risk.trailing_stop_distance_pct = distance_value
        except ValueError:
            pass
    
    # Загружаем общие настройки приложения из .env
    timeframe = os.getenv("TIMEFRAME", "").strip()
    if timeframe:
        settings.timeframe = timeframe
    
    leverage = os.getenv("LEVERAGE", "").strip()
    if leverage:
        try:
            settings.leverage = int(leverage)
        except ValueError:
            pass
    
    live_poll_seconds = os.getenv("LIVE_POLL_SECONDS", "").strip()
    if live_poll_seconds:
        try:
            settings.live_poll_seconds = int(live_poll_seconds)
        except ValueError:
            pass
    
    kline_limit = os.getenv("KLINE_LIMIT", "").strip()
    if not kline_limit:
        kline_limit = cleaned_env_values.get("KLINE_LIMIT", "")
    if kline_limit:
        try:
            settings.kline_limit = int(kline_limit)
            print(f"[config] KLINE_LIMIT loaded from .env: {settings.kline_limit}")
        except ValueError:
            pass
    
    return settings

