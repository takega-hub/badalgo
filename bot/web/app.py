"""
Flask веб-приложение для админки торгового бота.
"""
import warnings
# Подавляем предупреждения scikit-learn ДО импорта библиотек
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*sklearn.*')
warnings.filterwarnings('ignore', message='.*parallel.*')
warnings.filterwarnings('ignore', message='.*delayed.*')

import json
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import pytz

from flask import Flask, render_template, jsonify, request, session, redirect, url_for
from flask_cors import CORS
from functools import wraps
import hashlib

import pandas as pd

# Matplotlib and Seaborn imports (удалены, не используются в продакшене)
MATPLOTLIB_AVAILABLE = False

from bot.config import load_settings, AppSettings, StrategyParams, RiskParams
from bot.exchange.bybit_client import BybitClient

# Конфигурация логирования
WEB_VERBOSE_LOGGING = os.getenv("WEB_VERBOSE_LOGGING", "false").lower() == "true"

def _web_log(message: str, always_show: bool = False):
    """Логирование веб-интерфейса с фильтром."""
    if always_show or WEB_VERBOSE_LOGGING:
        print(message)
from bot.live import _get_balance, _get_position, _get_open_orders, _timeframe_to_bybit_interval, run_live_from_api
from bot.web.history import get_trades, get_signals, get_strategy_stats, get_smc_history
from bot.indicators import prepare_with_indicators
from bot.strategy import Action, build_signals, enrich_for_strategy, detect_market_phase, MarketPhase
from bot.multi_symbol_manager import MultiSymbolManager


app = Flask(__name__)
CORS(app)

# Настройка секретного ключа для сессий (можно задать через переменную окружения)
from datetime import datetime as dt
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'change-this-secret-key-in-production-' + hashlib.sha256(str(dt.now()).encode()).hexdigest()[:32])

# Отключаем логирование HTTP-запросов от werkzeug
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Настройки аутентификации (можно задать через переменные окружения)
ADMIN_USERNAME = os.getenv('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'admin123')  # ВАЖНО: измените в production!


def login_required(f):
    """Декоратор для проверки аутентификации."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session or not session.get('logged_in'):
            if request.path.startswith('/api/'):
                return jsonify({"error": "Authentication required"}), 401
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Функция для поиска модели для символа
def _find_model_for_symbol(symbol: str) -> Optional[str]:
    """Ищет обученную модель для указанного символа."""
    try:
        models_dir = Path(__file__).parent.parent.parent / "ml_models"
        if not models_dir.exists():
            return None
        
        # Ищем модели для этого символа
        # Формат: rf_SYMBOL_INTERVAL.pkl или xgb_SYMBOL_INTERVAL.pkl
        for model_file in models_dir.glob(f"*_{symbol}_*.pkl"):
            # Проверяем, что это действительно модель (не просто файл с таким именем)
            if model_file.is_file():
                return str(model_file)
        
        return None
    except Exception as e:
        print(f"[web] Error finding model for {symbol}: {e}")
        return None

# Функция для сохранения настроек в .env файл
def _save_symbol_settings_to_env(active_symbols: list, primary_symbol: str):
    """Сохраняет настройки активных символов в .env файл."""
    try:
        env_path = Path(__file__).parent.parent.parent / ".env"
        
        # Читаем существующий .env файл
        env_lines = []
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8') as f:
                env_lines = f.readlines()
        
        # Создаем новый файл, пропуская старые значения ACTIVE_SYMBOLS и PRIMARY_SYMBOL
        keys_to_update = {'ACTIVE_SYMBOLS', 'PRIMARY_SYMBOL'}
        
        with open(env_path, 'w', encoding='utf-8') as f:
            symbol_section_found = False
            for line in env_lines:
                line_stripped = line.strip()
                # Пропускаем строки с настройками символов
                if line_stripped and not line_stripped.startswith('#') and '=' in line_stripped:
                    key = line_stripped.split('=', 1)[0].strip()
                    if key in keys_to_update:
                        continue  # Пропускаем старые значения
                
                # Пропускаем комментарий о символах если он есть
                if "Multi-Symbol Trading" in line or "ACTIVE_SYMBOLS" in line or "PRIMARY_SYMBOL" in line:
                    if line.strip().startswith('#'):
                        symbol_section_found = True
                        continue
                
                f.write(line)
            
            # Добавляем настройки символов в конец
            if not symbol_section_found:
                f.write(f"\n# Multi-Symbol Trading (auto-updated by admin panel)\n")
            f.write(f"ACTIVE_SYMBOLS={','.join(active_symbols)}\n")
            f.write(f"PRIMARY_SYMBOL={primary_symbol}\n")
        
        print(f"[web] Symbol settings saved to .env: ACTIVE_SYMBOLS={','.join(active_symbols)}, PRIMARY_SYMBOL={primary_symbol}")
    
    except Exception as e:
        print(f"[web] Error saving symbol settings to .env: {e}")
        import traceback
        traceback.print_exc()


def _save_settings_to_env(settings: AppSettings):
    """Сохраняет настройки стратегий в .env файл."""
    try:
        env_path = Path(__file__).parent.parent.parent / ".env"
        
        # Читаем существующий .env файл
        env_lines = []
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8') as f:
                env_lines = f.readlines()
        
        # Создаем словарь существующих переменных
        env_dict = {}
        keys_to_update = {
            'ENABLE_TREND_STRATEGY', 'ENABLE_FLAT_STRATEGY', 'ENABLE_ML_STRATEGY', 
            'ENABLE_MOMENTUM_STRATEGY', 
            'ENABLE_LIQUIDITY_SWEEP_STRATEGY',
            'ENABLE_SMC_STRATEGY',
            'TRADING_SYMBOL', 'STRATEGY_PRIORITY',  # Приоритет стратегии
            'ML_CONFIDENCE_THRESHOLD', 'ML_MIN_SIGNAL_STRENGTH', 'ML_STABILITY_FILTER',
            'ML_MODEL_TYPE_FOR_ALL',  # Тип модели для всех пар
            # Strategy Parameters (Trend/Flat)
            'ADX_THRESHOLD', 'ADX_LENGTH', 'DI_LENGTH', 'BREAKOUT_LOOKBACK', 'BREAKOUT_VOLUME_MULT',
            'MOMENTUM_ADX_THRESHOLD',
            'SMA_LENGTH', 'PULLBACK_TOLERANCE', 'VOLUME_SPIKE_MULT', 'CONSOLIDATION_BARS', 'CONSOLIDATION_RANGE_PCT',
            'RSI_LENGTH', 'RSI_FLOOR', 'RSI_CEILING',
            # Range Strategy Parameters
            'BB_LENGTH', 'BB_STD', 'RANGE_RSI_OVERSOLD', 'RANGE_RSI_OVERBOUGHT', 'RANGE_VOLUME_MULT',
            'RANGE_TP_AGGRESSIVE', 'RANGE_STOP_LOSS_PCT', 'RANGE_BB_TOUCH_TOLERANCE_PCT',
            # Risk Management параметры
            'MAX_POSITION_USD', 'BASE_ORDER_USD', 'ADD_ORDER_USD', 'STOP_LOSS_PCT', 'TAKE_PROFIT_PCT',
            'BALANCE_PERCENT_PER_TRADE',
            # App Settings параметры
            'TIMEFRAME', 'LEVERAGE', 'LIVE_POLL_SECONDS'
        }
        
        # Собираем все существующие переменные, пропуская те, которые будем обновлять
        for line in env_lines:
            line_stripped = line.strip()
            if line_stripped and not line_stripped.startswith('#') and '=' in line_stripped:
                key, value = line_stripped.split('=', 1)
                key = key.strip()
                # Пропускаем ключи, которые будем обновлять (чтобы избежать дубликатов)
                if key not in keys_to_update:
                    env_dict[key] = value.strip()
        # Обновляем настройки стратегий
        env_dict['ENABLE_TREND_STRATEGY'] = str(settings.enable_trend_strategy).lower()
        env_dict['ENABLE_FLAT_STRATEGY'] = str(settings.enable_flat_strategy).lower()
        env_dict['ENABLE_ML_STRATEGY'] = str(settings.enable_ml_strategy).lower()
        env_dict['ENABLE_MOMENTUM_STRATEGY'] = str(settings.enable_momentum_strategy).lower()
        env_dict['ENABLE_LIQUIDITY_SWEEP_STRATEGY'] = str(settings.enable_liquidity_sweep_strategy).lower()
        env_dict['ENABLE_SMC_STRATEGY'] = str(settings.enable_smc_strategy).lower()
        env_dict['STRATEGY_PRIORITY'] = str(settings.strategy_priority).lower()
        env_dict['TRADING_SYMBOL'] = str(settings.symbol)
        
        # Сохраняем настройки ML стратегии
        env_dict['ML_CONFIDENCE_THRESHOLD'] = str(settings.ml_confidence_threshold)
        env_dict['ML_MIN_SIGNAL_STRENGTH'] = str(settings.ml_min_signal_strength)
        env_dict['ML_STABILITY_FILTER'] = str(settings.ml_stability_filter).lower()
        if settings.ml_model_type_for_all:
            env_dict['ML_MODEL_TYPE_FOR_ALL'] = str(settings.ml_model_type_for_all).lower()
        else:
            # Если None, удаляем из .env (очищаем настройку)
            env_dict.pop('ML_MODEL_TYPE_FOR_ALL', None)
        
        # Сохраняем параметры стратегии (Trend/Flat)
        env_dict['ADX_THRESHOLD'] = str(settings.strategy.adx_threshold)
        env_dict['ADX_LENGTH'] = str(settings.strategy.adx_length)
        env_dict['DI_LENGTH'] = str(settings.strategy.di_length)
        env_dict['BREAKOUT_LOOKBACK'] = str(settings.strategy.breakout_lookback)
        env_dict['BREAKOUT_VOLUME_MULT'] = str(settings.strategy.breakout_volume_mult)
        env_dict['MOMENTUM_ADX_THRESHOLD'] = str(settings.strategy.momentum_adx_threshold)
        env_dict['SMA_LENGTH'] = str(settings.strategy.sma_length)
        env_dict['PULLBACK_TOLERANCE'] = str(settings.strategy.pullback_tolerance)
        env_dict['VOLUME_SPIKE_MULT'] = str(settings.strategy.volume_spike_mult)
        env_dict['CONSOLIDATION_BARS'] = str(settings.strategy.consolidation_bars)
        env_dict['CONSOLIDATION_RANGE_PCT'] = str(settings.strategy.consolidation_range_pct)
        env_dict['RSI_LENGTH'] = str(settings.strategy.rsi_length)
        env_dict['RSI_FLOOR'] = str(settings.strategy.rsi_floor)
        env_dict['RSI_CEILING'] = str(settings.strategy.rsi_ceiling)
        
        # Сохраняем параметры Range стратегии
        env_dict['BB_LENGTH'] = str(settings.strategy.bb_length)
        env_dict['BB_STD'] = str(settings.strategy.bb_std)
        env_dict['RANGE_RSI_OVERSOLD'] = str(settings.strategy.range_rsi_oversold)
        env_dict['RANGE_RSI_OVERBOUGHT'] = str(settings.strategy.range_rsi_overbought)
        env_dict['RANGE_VOLUME_MULT'] = str(settings.strategy.range_volume_mult)
        env_dict['RANGE_TP_AGGRESSIVE'] = str(settings.strategy.range_tp_aggressive).lower()
        env_dict['RANGE_STOP_LOSS_PCT'] = str(settings.strategy.range_stop_loss_pct)
        env_dict['RANGE_BB_TOUCH_TOLERANCE_PCT'] = str(settings.strategy.range_bb_touch_tolerance_pct)
        
        # Сохраняем параметры управления рисками
        env_dict['MAX_POSITION_USD'] = str(settings.risk.max_position_usd)
        env_dict['BASE_ORDER_USD'] = str(settings.risk.base_order_usd)
        env_dict['ADD_ORDER_USD'] = str(settings.risk.add_order_usd)
        env_dict['STOP_LOSS_PCT'] = str(settings.risk.stop_loss_pct)
        env_dict['TAKE_PROFIT_PCT'] = str(settings.risk.take_profit_pct)
        env_dict['BALANCE_PERCENT_PER_TRADE'] = str(settings.risk.balance_percent_per_trade)
        
        # Сохраняем общие настройки приложения
        env_dict['TIMEFRAME'] = str(settings.timeframe)
        env_dict['LEVERAGE'] = str(settings.leverage)
        env_dict['LIVE_POLL_SECONDS'] = str(settings.live_poll_seconds)
        
        # Сохраняем обратно в .env
        with open(env_path, 'w', encoding='utf-8') as f:
            # Сохраняем все существующие строки (кроме обновляемых)
            strategy_section_found = False
            for line in env_lines:
                line_stripped = line.strip()
                # Пропускаем строки с настройками стратегий
                if line_stripped and not line_stripped.startswith('#') and '=' in line_stripped:
                    key = line_stripped.split('=', 1)[0].strip()
                    if key in keys_to_update:
                        continue  # Пропускаем старые значения
                
                # Пропускаем комментарий о стратегиях если он есть
                if "Strategy settings" in line or "auto-updated by admin panel" in line:
                    strategy_section_found = True
                    continue
                
                f.write(line)
            
            # Добавляем обновленные настройки в конец
            if not strategy_section_found:
                f.write(f"\n# Strategy settings (auto-updated by admin panel)\n")
            f.write(f"ENABLE_TREND_STRATEGY={env_dict['ENABLE_TREND_STRATEGY']}\n")
            f.write(f"ENABLE_FLAT_STRATEGY={env_dict['ENABLE_FLAT_STRATEGY']}\n")
            f.write(f"ENABLE_ML_STRATEGY={env_dict['ENABLE_ML_STRATEGY']}\n")
            f.write(f"ENABLE_MOMENTUM_STRATEGY={env_dict['ENABLE_MOMENTUM_STRATEGY']}\n")
            f.write(f"ENABLE_LIQUIDITY_SWEEP_STRATEGY={env_dict['ENABLE_LIQUIDITY_SWEEP_STRATEGY']}\n")
            f.write(f"ENABLE_SMC_STRATEGY={env_dict['ENABLE_SMC_STRATEGY']}\n")
            f.write(f"STRATEGY_PRIORITY={env_dict['STRATEGY_PRIORITY']}\n")
            f.write(f"TRADING_SYMBOL={env_dict['TRADING_SYMBOL']}\n")
            
            # Добавляем настройки ML стратегии
            f.write(f"\n# ML Strategy settings (auto-updated by admin panel)\n")
            f.write(f"ML_CONFIDENCE_THRESHOLD={env_dict['ML_CONFIDENCE_THRESHOLD']}\n")
            f.write(f"ML_MIN_SIGNAL_STRENGTH={env_dict['ML_MIN_SIGNAL_STRENGTH']}\n")
            f.write(f"ML_STABILITY_FILTER={env_dict['ML_STABILITY_FILTER']}\n")
            if 'ML_MODEL_TYPE_FOR_ALL' in env_dict:
                f.write(f"ML_MODEL_TYPE_FOR_ALL={env_dict['ML_MODEL_TYPE_FOR_ALL']}\n")
            
            # Добавляем параметры стратегии (Trend/Flat)
            f.write(f"\n# Strategy Parameters - Trend/Flat (auto-updated by admin panel)\n")
            f.write(f"ADX_THRESHOLD={env_dict['ADX_THRESHOLD']}\n")
            f.write(f"ADX_LENGTH={env_dict['ADX_LENGTH']}\n")
            f.write(f"DI_LENGTH={env_dict['DI_LENGTH']}\n")
            f.write(f"BREAKOUT_LOOKBACK={env_dict['BREAKOUT_LOOKBACK']}\n")
            f.write(f"BREAKOUT_VOLUME_MULT={env_dict['BREAKOUT_VOLUME_MULT']}\n")
            f.write(f"SMA_LENGTH={env_dict['SMA_LENGTH']}\n")
            f.write(f"PULLBACK_TOLERANCE={env_dict['PULLBACK_TOLERANCE']}\n")
            f.write(f"VOLUME_SPIKE_MULT={env_dict['VOLUME_SPIKE_MULT']}\n")
            f.write(f"CONSOLIDATION_BARS={env_dict['CONSOLIDATION_BARS']}\n")
            f.write(f"CONSOLIDATION_RANGE_PCT={env_dict['CONSOLIDATION_RANGE_PCT']}\n")
            f.write(f"RSI_LENGTH={env_dict['RSI_LENGTH']}\n")
            f.write(f"RSI_FLOOR={env_dict['RSI_FLOOR']}\n")
            f.write(f"RSI_CEILING={env_dict['RSI_CEILING']}\n")
            
            # Добавляем параметры Range стратегии
            f.write(f"\n# Strategy Parameters - Range (Mean Reversion) (auto-updated by admin panel)\n")
            f.write(f"BB_LENGTH={env_dict['BB_LENGTH']}\n")
            f.write(f"BB_STD={env_dict['BB_STD']}\n")
            f.write(f"RANGE_RSI_OVERSOLD={env_dict['RANGE_RSI_OVERSOLD']}\n")
            f.write(f"RANGE_RSI_OVERBOUGHT={env_dict['RANGE_RSI_OVERBOUGHT']}\n")
            f.write(f"RANGE_VOLUME_MULT={env_dict['RANGE_VOLUME_MULT']}\n")
            f.write(f"RANGE_TP_AGGRESSIVE={env_dict['RANGE_TP_AGGRESSIVE']}\n")
            f.write(f"RANGE_STOP_LOSS_PCT={env_dict['RANGE_STOP_LOSS_PCT']}\n")
            
            # Добавляем параметры управления рисками
            f.write(f"\n# Risk Management settings (auto-updated by admin panel)\n")
            f.write(f"MAX_POSITION_USD={env_dict['MAX_POSITION_USD']}\n")
            f.write(f"BASE_ORDER_USD={env_dict['BASE_ORDER_USD']}\n")
            f.write(f"ADD_ORDER_USD={env_dict['ADD_ORDER_USD']}\n")
            f.write(f"STOP_LOSS_PCT={env_dict['STOP_LOSS_PCT']}\n")
            f.write(f"TAKE_PROFIT_PCT={env_dict['TAKE_PROFIT_PCT']}\n")
            f.write(f"BALANCE_PERCENT_PER_TRADE={env_dict['BALANCE_PERCENT_PER_TRADE']}\n")
            
            # Добавляем общие настройки приложения
            f.write(f"\n# App Settings (auto-updated by admin panel)\n")
            f.write(f"TIMEFRAME={env_dict['TIMEFRAME']}\n")
            f.write(f"LEVERAGE={env_dict['LEVERAGE']}\n")
            f.write(f"LIVE_POLL_SECONDS={env_dict['LIVE_POLL_SECONDS']}\n")
        
    except Exception as e:
        print(f"[web] Error saving settings to .env: {e}")
        import traceback
        traceback.print_exc()

# Глобальные переменные для хранения состояния бота
bot_state = {
    "is_running": False,
    "last_update": None,
    "current_status": "Stopped",  # Stopped, Starting, Running, Analyzing, Signal Found, Order Placed, Error
    "current_phase": None,  # trend, flat
    "current_adx": None,
    "last_action": None,  # Последнее действие бота
    "last_action_time": None,
    "last_signal": None,  # Последний сигнал
    "last_signal_time": None,
    "last_error": None,
    "last_error_time": None,
    "trades_history": [],
    "signals_history": [],
}

# Поток для запуска бота (для обратной совместимости, используется только если MultiSymbolManager не инициализирован)
bot_thread: Optional[threading.Thread] = None
bot_stop_event = threading.Event()

# Загружаем настройки
settings: Optional[AppSettings] = None
client: Optional[BybitClient] = None

# MultiSymbolManager для управления несколькими торговыми парами
multi_symbol_manager: Optional[MultiSymbolManager] = None


def init_app():
    """Инициализация приложения."""
    global settings, client, multi_symbol_manager
    try:
        from bot.shared_settings import set_settings
        settings = load_settings()
        set_settings(settings)  # Сохраняем в shared_settings для работающего бота
        client = BybitClient(settings.api)
        
        # Инициализируем MultiSymbolManager для управления несколькими торговыми парами
        multi_symbol_manager = MultiSymbolManager(settings)
        print(f"[web] MultiSymbolManager initialized with active symbols: {settings.active_symbols}")
    except Exception as e:
        print(f"[web] Error in init_app: {e}")
        import traceback
        traceback.print_exc()
        # Не прерываем выполнение, но логируем ошибку


@app.before_request
def ensure_initialized():
    """Убедиться, что приложение инициализировано перед обработкой запросов."""
    # Объявляем global в самом начале функции
    global settings, client, multi_symbol_manager
    
    # Пропускаем статические файлы и страницу логина
    if request.path.startswith('/static/') or request.path == '/login':
        return
    
    # Пропускаем API запросы, которые могут быть проверены отдельно
    if request.path.startswith('/api/'):
        # Для API запросов мы проверяем инициализацию внутри самих обработчиков
        # Но все равно пытаемся инициализировать, если не инициализировано
        if settings is None or client is None:
            print("[web] Warning: App not initialized for API request, initializing now...")
            try:
                init_app()
            except Exception as e:
                print(f"[web] Error initializing app in before_request: {e}")
                import traceback
                traceback.print_exc()
                # Для API запросов возвращаем ошибку
                from flask import jsonify
                return jsonify({
                    "success": False,
                    "error": f"Application not initialized: {str(e)}"
                }), 500
        return
    
    if settings is None or client is None:
        print("[web] Warning: App not initialized, initializing now...")
        try:
            init_app()
        except Exception as e:
            print(f"[web] Error initializing app in before_request: {e}")
            import traceback
            traceback.print_exc()
            # Продолжаем выполнение, чтобы не блокировать все запросы


@app.route("/login", methods=["GET", "POST"])
def login():
    """Страница входа."""
    if request.method == "POST":
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['logged_in'] = True
            session['username'] = username
            if request.is_json:
                return jsonify({"success": True, "message": "Login successful"})
            return redirect(url_for('index'))
        else:
            if request.is_json:
                return jsonify({"error": "Invalid username or password"}), 401
            return render_template("login.html", error="Invalid username or password")
    
    # Если уже авторизован, перенаправляем на главную
    if session.get('logged_in'):
        return redirect(url_for('index'))
    
    return render_template("login.html")


@app.route("/logout", methods=["POST"])
def logout():
    """Выход из системы."""
    session.clear()
    if request.is_json:
        return jsonify({"success": True, "message": "Logged out successfully"})
    return redirect(url_for('login'))


@app.route("/")
@login_required
def index():
    """Главная страница со статистикой."""
    return render_template("index.html")


@app.route("/api/symbols/list")
@login_required
def api_symbols_list():
    """Получить список всех доступных символов."""
    available_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    return jsonify({
        "success": True,
        "symbols": available_symbols
    })


@app.route("/api/symbols/active")
@login_required
def api_symbols_active():
    """Получить список активных символов и основной символ."""
    if not settings:
        return jsonify({"error": "Settings not loaded"}), 500
    
    return jsonify({
        "success": True,
        "available_symbols": settings.symbols,
        "active_symbols": settings.active_symbols,
        "primary_symbol": settings.primary_symbol
    })


@app.route("/api/symbols/set-active", methods=["POST"])
@login_required
def api_symbols_set_active():
    """Установить активные символы и основной символ."""
    global settings, multi_symbol_manager, client
    
    if not settings:
        return jsonify({"error": "Settings not loaded"}), 500
    
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Получаем активные символы
        active_symbols = data.get("symbols", [])
        if not isinstance(active_symbols, list):
            return jsonify({"error": "symbols must be a list"}), 400
        
        # Получаем основной символ
        primary_symbol = data.get("primary", None)
        
        # Валидация символов
        available_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        active_symbols = [s.strip().upper() for s in active_symbols if s.strip().upper() in available_symbols]
        
        if not active_symbols:
            return jsonify({"error": "At least one active symbol is required"}), 400
        
        # Проверяем основной символ
        if primary_symbol:
            primary_symbol = primary_symbol.strip().upper()
            if primary_symbol not in available_symbols:
                return jsonify({"error": f"Invalid primary symbol: {primary_symbol}"}), 400
            if primary_symbol not in active_symbols:
                return jsonify({"error": f"Primary symbol {primary_symbol} must be in active symbols"}), 400
        else:
            # Если основной символ не указан, используем первый активный
            primary_symbol = active_symbols[0]
        
        # Обновляем настройки
        settings.active_symbols = active_symbols
        settings.primary_symbol = primary_symbol
        settings.symbol = primary_symbol  # Для обратной совместимости
        
        # Сохраняем в .env
        _save_symbol_settings_to_env(active_symbols, primary_symbol)
        
        # Обновляем MultiSymbolManager (в фоновом потоке, чтобы не блокировать Flask)
        if multi_symbol_manager:
            def update_and_restart():
                """Обновление настроек и перезапуск в фоновом потоке."""
                global multi_symbol_manager
                try:
                    print("[web] [background] Updating MultiSymbolManager settings from api_symbols_set_active...")
                    multi_symbol_manager.update_settings(settings)
                    print("[web] [background] Settings updated successfully")
                    
                    # Если менеджер уже запущен, перезапускаем его с новыми настройками
                    if multi_symbol_manager.running:
                        print(f"[web] [background] Restarting MultiSymbolManager with new active symbols: {active_symbols}")
                        multi_symbol_manager.stop()
                        # Даем время на остановку
                        import time
                        time.sleep(2)
                        multi_symbol_manager.start()
                        print("[web] [background] MultiSymbolManager restarted successfully")
                except Exception as e:
                    print(f"[web] [background] Error updating/restarting MultiSymbolManager: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Запускаем в фоновом потоке, чтобы не блокировать Flask
            import threading
            update_thread = threading.Thread(
                target=update_and_restart,
                name="SymbolSettingsUpdateThread",
                daemon=True
            )
            update_thread.start()
            print("[web] Settings update thread launched (won't block server)")
        
        # Обновляем shared_settings
        from bot.shared_settings import set_settings
        set_settings(settings)
        
        print(f"[web] Active symbols updated: {active_symbols}, primary: {primary_symbol}")
        
        return jsonify({
            "success": True,
            "message": f"Active symbols updated: {', '.join(active_symbols)}",
            "active_symbols": active_symbols,
            "primary_symbol": primary_symbol
        })
    
    except Exception as e:
        print(f"[web] Error setting active symbols: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/status")
@login_required
def api_status():
    """Получить текущий статус бота и аккаунта."""
    global multi_symbol_manager, bot_state, client, settings
    
    try:
        # Логируем запрос для диагностики
        # Проверяем инициализацию
        if not client:
            try:
                init_app()
                if not client:
                    return jsonify({
                        "error": "Client not initialized",
                        "bot_status": {
                            "is_running": False,
                            "current_status": "Error",
                            "last_error": "Client not initialized"
                        }
                    }), 500
            except Exception as e:
                print(f"[web] [api_status] ❌ Error initializing client: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    "error": f"Failed to initialize client: {str(e)}",
                    "bot_status": {
                        "is_running": False,
                        "current_status": "Error",
                        "last_error": f"Failed to initialize client: {str(e)}"
                    }
                }), 500
        
        if not settings:
            try:
                from bot.config import load_settings
                settings = load_settings()
                if not settings:
                    return jsonify({
                        "error": "Settings not loaded",
                        "bot_status": {
                            "is_running": False,
                            "current_status": "Error",
                            "last_error": "Settings not loaded"
                        }
                    }), 500
            except Exception as e:
                print(f"[web] [api_status] ❌ Error loading settings: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    "error": f"Failed to load settings: {str(e)}",
                    "bot_status": {
                        "is_running": False,
                        "current_status": "Error",
                        "last_error": f"Failed to load settings: {str(e)}"
                    }
                }), 500
        
        # Проверяем, что bot_state инициализирован
        if not bot_state:
            bot_state = {
                "is_running": False,
                "current_status": "Stopped",
                "last_update": None,
            }
        
        # Получаем символ из query параметра или используем primary_symbol
        symbol_from_query = request.args.get("symbol", None)
        if symbol_from_query and symbol_from_query in (settings.active_symbols if settings.active_symbols else ["BTCUSDT", "ETHUSDT", "SOLUSDT"]):
            primary_symbol = symbol_from_query
        else:
            # Получаем основной символ для отображения
            primary_symbol = settings.primary_symbol if settings.primary_symbol else (settings.symbol if settings.symbol else "BTCUSDT")
        
        try:
            balance = _get_balance(client)
        except Exception as e:
            print(f"[web] Error getting balance: {e}")
            balance = None
        
        try:
            position = _get_position(client, primary_symbol)
        except Exception as e:
            print(f"[web] Error getting position for {primary_symbol}: {e}")
            position = None
        
        try:
            orders = _get_open_orders(client, primary_symbol)
        except Exception as e:
            print(f"[web] Error getting open orders for {primary_symbol}: {e}")
            orders = []
        
        # Получаем статусы для всех активных символов, если используется MultiSymbolManager
        # ОПТИМИЗАЦИЯ: Быстро получаем статус воркеров, затем последовательно получаем позиции с обработкой ошибок
        symbols_status = {}
        if multi_symbol_manager and settings and settings.active_symbols:
            try:
                # Получаем статус воркеров (должно быть быстро, так как не требует API вызовов)
                all_workers_status = multi_symbol_manager.get_all_workers_status()
                
                # Получаем статусы для каждого символа с обработкой ошибок и таймаутом
                for symbol in settings.active_symbols:
                    try:
                        # Получаем worker статус (быстро)
                        worker_status = all_workers_status.get(symbol, {})
                        
                        # Получаем позицию и ордера с обработкой ошибок (может быть медленно)
                        symbol_position = None
                        symbol_orders = []
                        
                        try:
                            symbol_position = _get_position(client, symbol)
                        except Exception as pos_error:
                            print(f"[web] Error getting position for {symbol}: {pos_error}")
                            # Продолжаем выполнение даже при ошибке получения позиции
                        
                        try:
                            symbol_orders = _get_open_orders(client, symbol)
                        except Exception as orders_error:
                            print(f"[web] Error getting orders for {symbol}: {orders_error}")
                            # Продолжаем выполнение даже при ошибке получения ордеров
                        
                        symbols_status[symbol] = {
                            "is_running": worker_status.get("is_running", False),
                            "status": worker_status.get("current_status", "Stopped"),
                            "current_status": worker_status.get("current_status", "Stopped"),
                            "current_phase": worker_status.get("current_phase"),
                            "current_adx": worker_status.get("current_adx"),
                            "last_action": worker_status.get("last_action"),
                            "last_action_time": worker_status.get("last_action_time"),
                            "last_signal": worker_status.get("last_signal"),
                            "last_signal_time": worker_status.get("last_signal_time"),
                            "last_error": worker_status.get("last_error"),
                            "position": (
                                {
                                    "side": symbol_position.get("side"),
                                    "size": float(symbol_position.get('size', 0)),
                                    "avg_price": float(symbol_position.get('avg_price', 0)),
                                    "mark_price": float(symbol_position.get('mark_price', 0)) if symbol_position.get('mark_price') else float(symbol_position.get('avg_price', 0)),
                                    "unrealised_pnl": float(symbol_position.get('unrealised_pnl', 0)),
                                    "take_profit": (symbol_position.get('take_profit') or symbol_position.get('takeProfit') or ''),
                                    "stop_loss": (symbol_position.get('stop_loss') or symbol_position.get('stopLoss') or ''),
                                    "leverage": symbol_position.get('leverage', settings.leverage),
                                    "entry_reason": None,  # Будет заполнено ниже
                                    "strategy_type": None,  # Будет заполнено ниже
                                }
                                if symbol_position and symbol_position.get("side") and (float(symbol_position.get('size', 0)) > 0)
                                else None
                            ),
                            "open_orders": len(symbol_orders) if symbol_orders else 0
                        }
                        
                        # Добавляем информацию о сигнале открытия позиции для этого символа
                        if symbols_status[symbol].get("position") and symbols_status[symbol]["position"].get("side"):
                            try:
                                from bot.web.history import _load_history
                                history = _load_history()
                                trades = history.get("trades", [])
                                
                                position_side = symbols_status[symbol]["position"].get("side", "").upper()
                                position_side_normalized = "long" if position_side in ("LONG", "BUY") else "short" if position_side in ("SHORT", "SELL") else None
                                
                                if position_side_normalized:
                                    # Ищем последнюю открытую сделку (без exit_time) для этого символа и направления
                                    open_trades = [
                                        t for t in trades
                                        if t.get("symbol", "").upper() == symbol.upper() and
                                        t.get("side", "").lower() == position_side_normalized and
                                        (not t.get("exit_time") or t.get("exit_time") == "" or t.get("exit_time") is None)
                                    ]
                                    
                                    if open_trades:
                                        # Берем последнюю открытую сделку (самую свежую по entry_time)
                                        open_trades.sort(key=lambda x: x.get("entry_time", ""), reverse=True)
                                        last_open_trade = open_trades[0]
                                        
                                        entry_reason = last_open_trade.get("entry_reason", "")
                                        strategy_type = last_open_trade.get("strategy_type", "unknown")
                                        
                                        if symbols_status[symbol].get("position"):
                                            symbols_status[symbol]["position"]["entry_reason"] = entry_reason
                                            symbols_status[symbol]["position"]["strategy_type"] = strategy_type
                            except Exception as e:
                                print(f"[web] Error getting entry signal info for {symbol}: {e}")
                    except Exception as e:
                        print(f"[web] Error getting status for {symbol}: {e}")
                        import traceback
                        traceback.print_exc()
                        # Возвращаем базовый статус даже при ошибке
                        symbols_status[symbol] = {
                            "is_running": False,
                            "status": "Error",
                            "last_error": str(e)[:100],  # Ограничиваем длину ошибки
                            "position": None,
                            "open_orders": 0
                        }
            except Exception as e:
                print(f"[web] Error getting workers status: {e}")
                import traceback
                traceback.print_exc()
                # Возвращаем базовые статусы даже при ошибке
                for symbol in settings.active_symbols:
                    if symbol not in symbols_status:
                        symbols_status[symbol] = {
                            "is_running": False,
                            "status": "Error",
                            "last_error": str(e)[:100],  # Ограничиваем длину ошибки
                            "position": None,
                            "open_orders": 0
                        }
        else:
            # Если MultiSymbolManager не используется, все равно создаем статусы для активных символов
            if settings and settings.active_symbols:
                for symbol in settings.active_symbols:
                    try:
                        symbol_position = _get_position(client, symbol)
                        symbols_status[symbol] = {
                            "is_running": bot_state.get("is_running", False),
                            "status": bot_state.get("current_status", "Stopped"),
                            "current_status": bot_state.get("current_status", "Stopped"),
                            "current_phase": bot_state.get("current_phase"),
                            "current_adx": bot_state.get("current_adx"),
                            "last_action": bot_state.get("last_action"),
                            "last_action_time": bot_state.get("last_action_time"),
                            "last_signal": bot_state.get("last_signal"),
                            "last_signal_time": bot_state.get("last_signal_time"),
                            "last_error": bot_state.get("last_error"),
                            "position": {
                                "side": symbol_position.get("side") if symbol_position else None,
                                "size": float(symbol_position.get('size', 0)) if symbol_position else 0,
                                "avg_price": float(symbol_position.get('avg_price', 0)) if symbol_position else 0,
                                "unrealised_pnl": float(symbol_position.get('unrealised_pnl', 0)) if symbol_position else 0,
                            } if symbol_position else None,
                            "open_orders": 0
                        }
                    except Exception as e:
                        print(f"[web] Error creating status for {symbol}: {e}")
                        symbols_status[symbol] = {
                            "is_running": False,
                            "status": "Error",
                            "last_error": str(e),
                            "position": None,
                            "open_orders": 0
                        }
        
        # Получаем последний сигнал из истории для выбранного символа
        last_signal_from_history_str = None
        last_signal_time_from_history = None
        try:
            signals = get_signals(limit=1, symbol_filter=primary_symbol)
            if signals and len(signals) > 0:
                last_signal_data = signals[0]
                # Формируем строку для отображения последнего сигнала
                signal_action = last_signal_data.get("action", "").upper()
                signal_strategy = last_signal_data.get("strategy_type", "").upper()
                signal_price = last_signal_data.get("price", 0)
                last_signal_from_history_str = f"{signal_action} ({signal_strategy}) @ ${signal_price:.2f}"
                
                # Получаем время сигнала
                signal_timestamp = last_signal_data.get("timestamp", None)
                if signal_timestamp:
                    if isinstance(signal_timestamp, str):
                        try:
                            # Парсим timestamp
                            if 'T' in signal_timestamp:
                                dt = datetime.fromisoformat(signal_timestamp.replace('Z', '+00:00'))
                            else:
                                dt = datetime.strptime(signal_timestamp, '%Y-%m-%d %H:%M:%S')
                                dt = dt.replace(tzinfo=timezone.utc)
                            # Конвертируем в MSK
                            msk_tz = pytz.timezone('Europe/Moscow')
                            dt_msk = dt.astimezone(msk_tz)
                            last_signal_time_from_history = dt_msk.strftime('%Y-%m-%d %H:%M:%S MSK')
                        except Exception:
                            last_signal_time_from_history = signal_timestamp
                    else:
                        last_signal_time_from_history = str(signal_timestamp)
        except Exception as e:
            print(f"[web] Error getting last signal from history: {e}")
        
        # Используем последний сигнал из истории, если bot_state не содержит сигнала или если сигнал из истории новее
        final_last_signal = bot_state.get("last_signal")
        final_last_signal_time = bot_state.get("last_signal_time")
        
        if last_signal_from_history_str:
            # Если в bot_state нет сигнала или сигнал из истории новее - используем его
            if not final_last_signal or (last_signal_time_from_history and last_signal_time_from_history > (final_last_signal_time or "")):
                final_last_signal = last_signal_from_history_str
                final_last_signal_time = last_signal_time_from_history
        
        # Формируем данные ответа
        response_data = {
            "bot_status": {
                "is_running": bot_state.get("is_running", False),
                "current_status": bot_state.get("current_status", "Stopped"),
                "current_phase": bot_state.get("current_phase"),
                "current_adx": bot_state.get("current_adx"),
                "last_action": bot_state.get("last_action"),
                "last_action_time": bot_state.get("last_action_time"),
                "last_signal": final_last_signal,
                "last_signal_time": final_last_signal_time,
                "last_error": bot_state.get("last_error"),
                "last_error_time": bot_state.get("last_error_time"),
                "last_update": bot_state.get("last_update"),
            },
            "account": {
                "balance": f"{balance:.2f} USDT" if balance is not None else "N/A",
                "primary_symbol": primary_symbol,
                "active_symbols": settings.active_symbols if settings else [],
                "leverage": settings.leverage,
            },
            "position": {
                "side": position.get("side") if position else None,
                "size": float(position.get('size', 0)) if position else 0,
                "avg_price": float(position.get('avg_price', 0)) if position else 0,
                "mark_price": float(position.get('mark_price', 0)) if position else (float(position.get('avg_price', 0)) if position else 0),
                "unrealised_pnl": float(position.get('unrealised_pnl', 0)) if position else 0,
                "take_profit": (position.get('take_profit') or position.get('takeProfit') or '') if position else '',
                "stop_loss": (position.get('stop_loss') or position.get('stopLoss') or '') if position else '',
                "leverage": position.get('leverage', settings.leverage) if position else settings.leverage,
                "entry_reason": None,  # Будет заполнено ниже
                "strategy_type": None,  # Будет заполнено ниже
            } if position else None,
            "open_orders": len(orders) if isinstance(orders, list) else 0,
            "symbols_status": symbols_status if symbols_status else None,
            "symbol": primary_symbol,  # Добавляем символ в ответ для использования в интерфейсе
        }
        
        # Добавляем информацию о сигнале открытия позиции, если позиция открыта
        if position and position.get("side") and float(position.get('size', 0)) > 0:
            try:
                from bot.web.history import _load_history
                history = _load_history()
                trades = history.get("trades", [])
                
                # Ищем открытую позицию (сделка без exit_time или с пустым exit_time)
                # для текущего символа и соответствующего направления
                position_side = position.get("side", "").upper()
                position_side_normalized = "long" if position_side in ("LONG", "BUY") else "short" if position_side in ("SHORT", "SELL") else None
                
                if position_side_normalized:
                    # Ищем последнюю открытую сделку (без exit_time) для этого символа и направления
                    open_trades = [
                        t for t in trades
                        if t.get("symbol", "").upper() == primary_symbol.upper() and
                        t.get("side", "").lower() == position_side_normalized and
                        (not t.get("exit_time") or t.get("exit_time") == "" or t.get("exit_time") is None)
                    ]
                    
                    if open_trades:
                        # Берем последнюю открытую сделку (самую свежую по entry_time)
                        open_trades.sort(key=lambda x: x.get("entry_time", ""), reverse=True)
                        last_open_trade = open_trades[0]
                        
                        entry_reason = last_open_trade.get("entry_reason", "")
                        strategy_type = last_open_trade.get("strategy_type", "unknown")
                        
                        if response_data.get("position"):
                            response_data["position"]["entry_reason"] = entry_reason
                            response_data["position"]["strategy_type"] = strategy_type
            except Exception as e:
                print(f"[web] Error getting entry signal info for {primary_symbol}: {e}")
        
        # Добавляем статистику PnL для выбранного символа
        try:
            from bot.web.history import get_pnl_stats
            response_data["pnl_stats"] = get_pnl_stats(symbol=primary_symbol)
        except Exception as e:
            print(f"[web] Error getting PnL stats for {primary_symbol}: {e}")
            response_data["pnl_stats"] = {
                "pnl_today": 0.0,
                "pnl_week": 0.0,
                "pnl_month": 0.0,
                "pnl_total": 0.0,
            }
        
        return jsonify(response_data)
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[web] [api_status] ❌ ERROR in api_status: {e}")
        print(f"[web] [api_status] Full traceback:\n{error_trace}")
        
        # Возвращаем минимальный ответ с ошибкой, чтобы фронтенд мог отобразить состояние
        # НЕ возвращаем 500, чтобы фронтенд мог обработать ошибку
        return jsonify({
            "error": str(e),
            "error_details": error_trace.split('\n')[-10:] if len(error_trace.split('\n')) > 10 else error_trace.split('\n'),
            "bot_status": {
                "is_running": bot_state.get("is_running", False) if bot_state else False,
                "current_status": "Error",
                "last_error": str(e),
                "last_update": datetime.now(timezone.utc).isoformat()
            },
            "account": {
                "balance": "N/A",
                "primary_symbol": settings.primary_symbol if settings and hasattr(settings, 'primary_symbol') else ("BTCUSDT" if settings else "BTCUSDT"),
                "active_symbols": settings.active_symbols if settings and hasattr(settings, 'active_symbols') else [],
                "leverage": settings.leverage if settings and hasattr(settings, 'leverage') else 10
            },
            "position": None,
            "open_orders": 0,
            "symbols_status": {}
        }), 200  # Возвращаем 200, чтобы фронтенд мог обработать ошибку в data.error

@app.route("/api/settings")
@login_required
def api_get_settings():
    """Получить текущие настройки."""
    if not settings:
        return jsonify({"error": "Settings not loaded"}), 500
    
    return jsonify({
        "strategy": {
            "adx_threshold": settings.strategy.adx_threshold,
            "adx_length": settings.strategy.adx_length,
            "di_length": settings.strategy.di_length,
            "breakout_lookback": settings.strategy.breakout_lookback,
            "breakout_volume_mult": settings.strategy.breakout_volume_mult,
            "sma_length": settings.strategy.sma_length,
            "pullback_tolerance": settings.strategy.pullback_tolerance,
            "volume_spike_mult": settings.strategy.volume_spike_mult,
            "consolidation_bars": settings.strategy.consolidation_bars,
            "consolidation_range_pct": settings.strategy.consolidation_range_pct,
            "rsi_length": settings.strategy.rsi_length,
            "rsi_floor": settings.strategy.rsi_floor,
            "rsi_ceiling": settings.strategy.rsi_ceiling,
            "bb_length": settings.strategy.bb_length,
            "bb_std": settings.strategy.bb_std,
            "range_rsi_oversold": settings.strategy.range_rsi_oversold,
            "range_rsi_overbought": settings.strategy.range_rsi_overbought,
            "range_volume_mult": settings.strategy.range_volume_mult,
            "range_tp_aggressive": settings.strategy.range_tp_aggressive,
            "range_stop_loss_pct": settings.strategy.range_stop_loss_pct,
        },
        "risk": {
            "max_position_usd": settings.risk.max_position_usd,
            "base_order_usd": settings.risk.base_order_usd,
            "add_order_usd": settings.risk.add_order_usd,
            "stop_loss_pct": settings.risk.stop_loss_pct,
            "take_profit_pct": settings.risk.take_profit_pct,
            "balance_percent_per_trade": settings.risk.balance_percent_per_trade,
            "enable_loss_cooldown": settings.risk.enable_loss_cooldown,
            "loss_cooldown_minutes": settings.risk.loss_cooldown_minutes,
            "max_consecutive_losses": settings.risk.max_consecutive_losses,
            "enable_trailing_stop": settings.risk.enable_trailing_stop,
            "trailing_stop_activation_pct": settings.risk.trailing_stop_activation_pct,
            "trailing_stop_distance_pct": settings.risk.trailing_stop_distance_pct,
            "enable_partial_close": settings.risk.enable_partial_close,
            "partial_close_pct": settings.risk.partial_close_pct,
            "partial_close_at_tp_pct": settings.risk.partial_close_at_tp_pct,
            "enable_profit_protection": settings.risk.enable_profit_protection,
            "profit_protection_activation_pct": settings.risk.profit_protection_activation_pct,
            "profit_protection_retreat_pct": settings.risk.profit_protection_retreat_pct,
            "enable_breakeven": settings.risk.enable_breakeven,
            "breakeven_activation_pct": settings.risk.breakeven_activation_pct,
            "enable_smart_add": settings.risk.enable_smart_add,
            "smart_add_pullback_pct": settings.risk.smart_add_pullback_pct,
            "enable_atr_entry_filter": settings.risk.enable_atr_entry_filter,
            "max_atr_progress_pct": settings.risk.max_atr_progress_pct,
        },
            "app": {
                "symbol": settings.symbol,
                "timeframe": settings.timeframe,
                "leverage": settings.leverage,
                "live_poll_seconds": settings.live_poll_seconds,
                "kline_limit": settings.kline_limit,
                "strategy_type": settings.strategy_type,
                "enable_trend_strategy": settings.enable_trend_strategy,
                "enable_flat_strategy": settings.enable_flat_strategy,
                "enable_ml_strategy": settings.enable_ml_strategy,
                "enable_momentum_strategy": settings.enable_momentum_strategy,
                "enable_liquidity_sweep_strategy": settings.enable_liquidity_sweep_strategy,
                "enable_smc_strategy": settings.enable_smc_strategy,
                "strategy_priority": settings.strategy_priority,
                "ml_model_path": settings.ml_model_path,
                "ml_model_type_for_all": settings.ml_model_type_for_all or "",
                "ml_confidence_threshold": settings.ml_confidence_threshold,
                "ml_min_signal_strength": settings.ml_min_signal_strength,
                "ml_stability_filter": settings.ml_stability_filter,
            },
        })


@app.route("/api/settings", methods=["POST"])
@login_required
def api_update_settings():
    """Обновить настройки."""
    global settings, client
    
    if not settings:
        return jsonify({"error": "Settings not loaded"}), 500
    
    data = request.json
    
    try:
        # Обновляем параметры стратегии
        if "strategy" in data:
            strategy_data = data["strategy"]
            for key, value in strategy_data.items():
                if hasattr(settings.strategy, key):
                    # Специальная обработка для boolean значений
                    if key == "range_tp_aggressive":
                        setattr(settings.strategy, key, bool(value))
                    # Специальная обработка для числовых значений (могут прийти с запятой как разделителем)
                    elif isinstance(value, str):
                        # Заменяем запятую на точку для правильного парсинга float
                        value_str = value.replace(',', '.')
                        try:
                            # Проверяем, является ли это числом
                            if '.' in value_str or 'e' in value_str.lower() or 'E' in value_str:
                                value = float(value_str)
                            else:
                                # Пробуем как int
                                try:
                                    value = int(value_str)
                                except ValueError:
                                    value = float(value_str)
                        except (ValueError, TypeError):
                            pass  # Оставляем исходное значение, если не удалось распарсить
                    setattr(settings.strategy, key, value)
        
        # Обновляем параметры риска
        if "risk" in data:
            risk_data = data["risk"]
            for key, value in risk_data.items():
                if hasattr(settings.risk, key):
                    # Обрабатываем числовые значения (могут прийти с запятой как разделителем)
                    if isinstance(value, str):
                        # Заменяем запятую на точку для правильного парсинга float
                        value_str = value.replace(',', '.')
                        try:
                            # Проверяем, является ли это числом
                            if '.' in value_str or 'e' in value_str.lower() or 'E' in value_str:
                                value = float(value_str)
                            else:
                                # Проверяем, является ли это boolean
                                if value_str.lower() in ('true', 'false', '1', '0', 'yes', 'no'):
                                    value = value_str.lower() in ('true', '1', 'yes')
                                else:
                                    # Пробуем как int, потом как float
                                    try:
                                        value = int(value_str)
                                    except ValueError:
                                        value = float(value_str)
                        except (ValueError, TypeError):
                            pass  # Оставляем исходное значение, если не удалось распарсить
                    setattr(settings.risk, key, value)
        
        # Обновляем параметры приложения
        if "app" in data:
            app_data = data["app"]
            for key, value in app_data.items():
                if hasattr(settings, key):
                    # Специальная обработка для boolean значений
                    if key in ("enable_trend_strategy", "enable_flat_strategy", "enable_ml_strategy", 
                               "enable_momentum_strategy", 
                               "enable_liquidity_sweep_strategy", "enable_smc_strategy", "ml_stability_filter"):
                        setattr(settings, key, bool(value))
                    elif key == "symbol":
                        # Проверяем доступные пары
                        available_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
                        if value in available_symbols:
                            setattr(settings, key, value)
                        else:
                            return jsonify({"error": f"Invalid symbol: {value}. Allowed: {', '.join(available_symbols)}"}), 400
                    elif key == "ml_min_signal_strength":
                        # Проверяем допустимые значения силы сигнала
                        allowed_strengths = ["слабое", "умеренное", "среднее", "сильное", "очень_сильное"]
                        if value in allowed_strengths:
                            setattr(settings, key, value)
                        else:
                            return jsonify({"error": f"Invalid ml_min_signal_strength: {value}. Allowed: {', '.join(allowed_strengths)}"}), 400
                    elif key == "ml_confidence_threshold":
                        # Проверяем диапазон (0-1)
                        try:
                            # Обрабатываем запятую как разделитель десятичных дробей (для локализованного ввода)
                            value_str = str(value).replace(',', '.')
                            threshold = float(value_str)
                            if 0 <= threshold <= 1:
                                setattr(settings, key, threshold)
                                print(f"[web] ML confidence threshold updated: {threshold}")
                            else:
                                return jsonify({"error": f"ml_confidence_threshold must be between 0 and 1, got {threshold}"}), 400
                        except (ValueError, TypeError) as e:
                            return jsonify({"error": f"Invalid ml_confidence_threshold: {value}. Must be a number between 0 and 1. Error: {e}"}), 400
                    elif key == "strategy_priority":
                        # Проверяем допустимые значения приоритета стратегии
                        allowed_priorities = ["trend", "flat", "ml", "momentum", "liquidity", "smc", "hybrid", "confluence"]
                        if value in allowed_priorities:
                            setattr(settings, key, value)
                            print(f"[web] Strategy priority updated: {value}")
                        else:
                            return jsonify({"error": f"Invalid strategy_priority: {value}. Allowed: {', '.join(allowed_priorities)}"}), 400
                    elif key == "ml_model_type_for_all":
                        # Проверяем допустимые значения типа модели для всех пар
                        allowed_types = ["rf", "xgb", "ensemble", ""]
                        # Безопасная обработка: проверяем None и пустые значения
                        if value is None or value == "":
                            # Если пустое значение, устанавливаем None (авто-выбор)
                            setattr(settings, key, None)
                            print(f"[web] ML model type for all pairs updated: None (auto)")
                        elif isinstance(value, str) and value.lower() in allowed_types:
                            # Если значение - строка и входит в допустимые, устанавливаем в нижнем регистре
                            setattr(settings, key, value.lower())
                            print(f"[web] ML model type for all pairs updated: {value.lower()}")
                        else:
                            return jsonify({"error": f"Invalid ml_model_type_for_all: {value}. Allowed: {', '.join([t for t in allowed_types if t])}, or empty for auto"}), 400
                    elif key in ("timeframe", "leverage", "live_poll_seconds"):
                        # Обрабатываем числовые значения для app параметров (могут прийти с запятой)
                        if isinstance(value, str):
                            value_str = value.replace(',', '.')
                            try:
                                if key == "leverage" or key == "live_poll_seconds":
                                    setattr(settings, key, int(float(value_str)))
                                else:
                                    setattr(settings, key, value_str)
                            except (ValueError, TypeError):
                                setattr(settings, key, value)  # Оставляем исходное значение, если не удалось распарсить
                        else:
                            setattr(settings, key, value)
                    else:
                        setattr(settings, key, value)
            
            # При изменении символа ищем модель автоматически
            if "symbol" in app_data:
                symbol = app_data["symbol"]
                old_model_path = settings.ml_model_path
                model_path = _find_model_for_symbol(symbol)
                if model_path:
                    settings.ml_model_path = model_path
                else:
                    settings.ml_model_path = None
        
        # Сохраняем настройки в .env файл после обновления (вне зависимости от секции)
        if "risk" in data or "app" in data or "strategy" in data:
            _save_settings_to_env(settings)
        
        # Перезагружаем настройки из .env чтобы они точно применились
        from dotenv import load_dotenv
        import os
        env_path = Path(__file__).parent.parent.parent / ".env"
        
        # Принудительно очищаем кэш переменных окружения (включая ML настройки, параметры стратегии, риска и app)
        env_keys_to_clear = [
            "TRADING_SYMBOL", "ENABLE_TREND_STRATEGY", "ENABLE_FLAT_STRATEGY", "ENABLE_ML_STRATEGY",
            "ENABLE_MOMENTUM_STRATEGY", 
            "ENABLE_LIQUIDITY_SWEEP_STRATEGY",
            "STRATEGY_PRIORITY",  # Приоритет стратегии
            "ML_CONFIDENCE_THRESHOLD", "ML_MIN_SIGNAL_STRENGTH", "ML_STABILITY_FILTER",
            # Strategy Parameters
            "ADX_THRESHOLD", "ADX_LENGTH", "DI_LENGTH", "BREAKOUT_LOOKBACK", "BREAKOUT_VOLUME_MULT",
            "SMA_LENGTH", "PULLBACK_TOLERANCE", "VOLUME_SPIKE_MULT", "CONSOLIDATION_BARS", "CONSOLIDATION_RANGE_PCT",
            "RSI_LENGTH", "RSI_FLOOR", "RSI_CEILING",
            "BB_LENGTH", "BB_STD", "RANGE_RSI_OVERSOLD", "RANGE_RSI_OVERBOUGHT", "RANGE_VOLUME_MULT",
            "RANGE_TP_AGGRESSIVE", "RANGE_STOP_LOSS_PCT",
            # Risk Management параметры
            "MAX_POSITION_USD", "BASE_ORDER_USD", "ADD_ORDER_USD", "STOP_LOSS_PCT", "TAKE_PROFIT_PCT",
            "BALANCE_PERCENT_PER_TRADE", "TIMEFRAME", "LEVERAGE", "LIVE_POLL_SECONDS"
        ]
        for key in env_keys_to_clear:
            if key in os.environ:
                del os.environ[key]
        
        # Загружаем .env с принудительным перезаписыванием
        load_dotenv(dotenv_path=env_path, override=True)
        
        # Перезагружаем настройки и обновляем глобальную переменную
        settings = load_settings()
        # Также обновляем клиент с новыми настройками
        client = BybitClient(settings.api)
        
        # Обновляем настройки в shared_settings для работающего бота
        from bot.shared_settings import set_settings
        set_settings(settings)
        
        print(f"[web] Settings updated. ML config: confidence={settings.ml_confidence_threshold}, strength={settings.ml_min_signal_strength}, stability={settings.ml_stability_filter}")
        print(f"[web] Strategy settings: TREND={settings.enable_trend_strategy}, FLAT={settings.enable_flat_strategy}, ML={settings.enable_ml_strategy}, MOMENTUM={settings.enable_momentum_strategy}, LIQUIDITY={settings.enable_liquidity_sweep_strategy}")
        
        # Возвращаем обновленные настройки в ответе, чтобы фронтенд мог их использовать
        # Используем settings после перезагрузки, чтобы гарантировать актуальные значения
        response_data = {
            "success": True, 
            "message": "Settings updated",
            "settings": {
                "symbol": settings.symbol,
                "enable_trend_strategy": settings.enable_trend_strategy,
                "enable_flat_strategy": settings.enable_flat_strategy,
                "enable_ml_strategy": settings.enable_ml_strategy,
                "enable_momentum_strategy": settings.enable_momentum_strategy,
                "enable_liquidity_sweep_strategy": settings.enable_liquidity_sweep_strategy,
                "enable_smc_strategy": settings.enable_smc_strategy,
                "strategy_priority": settings.strategy_priority,
                "ml_model_path": settings.ml_model_path,
                "ml_model_type_for_all": settings.ml_model_type_for_all or "",
            }
        }
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/trades")
@login_required
def api_trades():
    """Получить историю сделок и статистику PnL."""
    strategy_filter = request.args.get("strategy", None)
    symbol_filter = request.args.get("symbol", None)
    
    # Если symbol не указан, используем текущий символ из настроек
    if not symbol_filter and settings:
        symbol_filter = settings.symbol
    
    # Если symbol не указан, используем primary_symbol из настроек
    if not symbol_filter and settings:
        symbol_filter = settings.primary_symbol if settings.primary_symbol else settings.symbol
    
    trades = get_trades(limit=100, strategy_filter=strategy_filter, symbol_filter=symbol_filter)
    
    # Получаем статистику PnL для выбранного символа
    from bot.web.history import get_pnl_stats
    pnl_stats = get_pnl_stats(symbol=symbol_filter)
    
    # Определяем, используется ли тестнет (по умолчанию тестнет согласно памяти пользователя)
    is_testnet = True
    if settings:
        base_url = settings.api.base_url
        is_testnet = "testnet" in base_url.lower() if base_url else True
    
    return jsonify({
        "trades": trades,
        "total": len(trades),
        "pnl_stats": pnl_stats,
        "symbol": symbol_filter or (settings.symbol if settings else "ALL"),
        "is_testnet": is_testnet,
    })


@app.route("/api/pnl/combined")
@login_required
def api_combined_pnl_stats():
    """Получить сводную статистику PnL по всем активным символам."""
    try:
        from bot.web.history import get_combined_pnl_stats, get_all_symbols_pnl_stats
        
        # Получаем активные символы из настроек
        active_symbols = settings.active_symbols if settings and settings.active_symbols else ["BTCUSDT"]
        
        # Получаем сводную статистику
        combined_stats = get_combined_pnl_stats(active_symbols)
        
        # Получаем статистику по каждому символу
        per_symbol_stats = get_all_symbols_pnl_stats(active_symbols)
        
        return jsonify({
            "success": True,
            "combined": combined_stats,
            "per_symbol": per_symbol_stats,
            "active_symbols": active_symbols,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/trades/remove-duplicates", methods=["POST"])
@login_required
def api_remove_duplicate_trades():
    """Удалить дубликаты сделок из истории."""
    try:
        from bot.web.history import remove_duplicate_trades
        removed_count = remove_duplicate_trades()
        return jsonify({
            "success": True,
            "removed_count": removed_count,
            "message": f"Removed {removed_count} duplicate trades"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/strategy/stats")
@login_required
def api_strategy_stats():
    """Получить статистику по стратегиям."""
    strategy_type = request.args.get("strategy", None)
    stats = get_strategy_stats(strategy_type=strategy_type)
    return jsonify(stats)


@app.route("/api/strategy/stats/all")
@login_required
def api_all_strategy_stats():
    """Получить статистику по всем стратегиям."""
    all_stats = {
        "trend": get_strategy_stats(strategy_type="trend"),
        "flat": get_strategy_stats(strategy_type="flat"),
        "ml": get_strategy_stats(strategy_type="ml"),
        "all": get_strategy_stats(strategy_type=None),
    }
    return jsonify(all_stats)


@app.route("/api/smc/history")
@login_required
def api_smc_history():
    """Получить историю сигналов SMC из CSV файла."""
    try:
        limit = request.args.get("limit", 100, type=int)
        smc_history = get_smc_history(limit=limit)
        
        return jsonify({
            "success": True,
            "history": smc_history,
            "total": len(smc_history),
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/ml/model/info")
@login_required
def api_ml_model_info():
    """Получить информацию о текущей ML модели."""
    if not settings:
        return jsonify({"error": "Settings not loaded"}), 500
    
    if not settings.ml_model_path:
        # Пытаемся найти модель автоматически для текущего символа
        model_path = _find_model_for_symbol(settings.symbol)
        if model_path:
            settings.ml_model_path = model_path
            print(f"[web] Auto-found ML model for {settings.symbol}: {model_path}")
        else:
            return jsonify({"error": "No ML model configured"}), 404
    
    if not settings.ml_model_path:
        return jsonify({"error": "No ML model configured"}), 404
    
    # Отложенный импорт для избежания deadlock при многопоточности
    try:
        from bot.ml.model_trainer import ModelTrainer
    except Exception as e:
        print(f"[web] Error importing ModelTrainer: {e}")
        return jsonify({"error": f"Failed to import ML trainer: {str(e)}"}), 500
    
    from pathlib import Path
    import os
    
    try:
        trainer = ModelTrainer()
        metadata = trainer.load_model_metadata(settings.ml_model_path)
    except Exception as e:
        print(f"[web] Error loading model metadata: {e}")
        return jsonify({"error": f"Failed to load model metadata: {str(e)}"}), 500
    
    # Если метаданных нет (старая модель), создаем их из имени файла и даты модификации
    if not metadata:
        model_path = Path(settings.ml_model_path)
        if model_path.exists():
            # Пытаемся извлечь информацию из имени файла: rf_ETHUSDT_15.pkl или ensemble_ETHUSDT_15.pkl
            filename = model_path.name
            parts = filename.replace('.pkl', '').split('_')
            symbol = parts[1] if len(parts) > 1 else settings.symbol
            interval = parts[2] if len(parts) > 2 else "15"
            model_type_from_filename = parts[0].lower() if len(parts) > 0 else "unknown"
            
            # Получаем дату модификации файла
            mtime = os.path.getmtime(settings.ml_model_path)
            from datetime import datetime
            trained_at = datetime.fromtimestamp(mtime).isoformat()
            
            # Пытаемся загрузить модель чтобы получить метрики
            try:
                with open(settings.ml_model_path, "rb") as f:
                    import pickle
                    model_data = pickle.load(f)
                    metrics = model_data.get("metrics", {})
                    metadata = {
                        "symbol": symbol,
                        "interval": interval,
                        "model_type": model_type_from_filename,
                        "trained_at": trained_at,
                        "accuracy": metrics.get("accuracy", 0.0),
                        "cv_mean": metrics.get("cv_mean", 0.0),
                        "cv_std": metrics.get("cv_std", 0.0),
                        "precision": metrics.get("precision", None),
                        "recall": metrics.get("recall", None),
                        "f1_score": metrics.get("f1_score", None),
                        "cv_f1_mean": metrics.get("cv_f1_mean", None),
                    }
            except Exception as e:
                print(f"[web] Error loading model data: {e}")
                # Создаем минимальные метаданные
                metadata = {
                    "symbol": symbol,
                    "interval": interval,
                    "model_type": model_type_from_filename,
                    "trained_at": trained_at,
                    "accuracy": 0.0,
                    "cv_mean": 0.0,
                    "cv_std": 0.0,
                }
        else:
            return jsonify({"error": f"Model file not found: {settings.ml_model_path}"}), 404
    
    if not metadata:
        return jsonify({"error": "Could not load or create model metadata"}), 500
    
    # Получаем статистику ML стратегии
    ml_stats = get_strategy_stats(strategy_type="ml")
    
    # Рекомендация переобучения
    should_retrain = False
    retrain_reasons = []
    
    if ml_stats["total_trades"] > 0:
        # Рекомендуем переобучение если:
        # 1. Win rate < 40%
        if ml_stats["win_rate"] < 40:
            should_retrain = True
            retrain_reasons.append(f"Low win rate: {ml_stats['win_rate']:.1f}%")
        
        # 2. Total PnL отрицательный и много сделок
        if ml_stats["total_pnl"] < 0 and ml_stats["total_trades"] > 10:
            should_retrain = True
            retrain_reasons.append(f"Negative PnL: {ml_stats['total_pnl']:.2f} USDT")
        
        # 3. Модель старая (больше 30 дней)
        if metadata.get("trained_at"):
            from datetime import datetime, timedelta
            trained_date = datetime.fromisoformat(metadata["trained_at"])
            days_old = (datetime.now() - trained_date).days
            if days_old > 30:
                should_retrain = True
                retrain_reasons.append(f"Model is {days_old} days old")
    
    # Определяем, является ли модель ансамблем
    model_type = metadata.get("model_type", "").lower() if metadata else ""
    is_ensemble = "ensemble" in model_type
    
    # Получаем дополнительные метрики для ансамбля
    ensemble_metrics = {}
    if is_ensemble and metadata:
        ensemble_metrics = {
            "precision": metadata.get("precision"),
            "recall": metadata.get("recall"),
            "f1_score": metadata.get("f1_score"),
            "cv_f1_mean": metadata.get("cv_f1_mean"),
            "rf_weight": metadata.get("rf_weight"),
            "xgb_weight": metadata.get("xgb_weight"),
            "ensemble_method": metadata.get("ensemble_method"),
        }
    
    return jsonify({
        "metadata": metadata,
        "stats": ml_stats,
        "should_retrain": should_retrain,
        "retrain_reasons": retrain_reasons,
        "current_model_path": settings.ml_model_path if settings and settings.ml_model_path else None,
        "is_ensemble": is_ensemble,
        "ensemble_metrics": ensemble_metrics,
    })


@app.route("/api/ml/models/all-pairs")
@login_required
def api_ml_models_all_pairs():
    """Получить информацию о выбранных моделях для всех пар."""
    if not settings:
        return jsonify({"error": "Settings not loaded"}), 500
    
    try:
        from pathlib import Path
        from bot.ml.model_trainer import ModelTrainer
        from bot.multi_symbol_manager import MultiSymbolManager
        
        # Получаем менеджер для доступа к настройкам для каждой пары
        # Создаем временный менеджер для получения информации о моделях
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        models_info = {}
        
        # Получаем предпочтение типа модели
        model_type_preference = getattr(settings, 'ml_model_type_for_all', None)
        
        models_dir = Path(__file__).parent.parent.parent / "ml_models"
        trainer = ModelTrainer()
        
        for symbol in symbols:
            # Ищем модель для символа
            found_model = None
            
            # Сначала проверяем, есть ли явно выбранная модель в settings.ml_model_path
            # и соответствует ли она текущему символу И типу модели (если ml_model_type_for_all задан)
            if settings.ml_model_path:
                model_path_obj = Path(settings.ml_model_path)
                if model_path_obj.exists():
                    # Извлекаем символ и тип модели из имени файла модели
                    model_filename = model_path_obj.name
                    # Формат: ensemble_BTCUSDT_15.pkl или rf_ETHUSDT_15.pkl
                    if "_" in model_filename:
                        parts = model_filename.replace('.pkl', '').split('_')
                        if len(parts) >= 2 and parts[1] == symbol:
                            # Модель соответствует текущему символу
                            model_type_from_filename = parts[0].lower()
                            if model_type_preference:
                                # Если задан глобальный тип модели, проверяем соответствие
                                if model_type_from_filename == model_type_preference.lower():
                                    found_model = str(model_path_obj)
                                    print(f"[web] Using explicitly selected model for {symbol}: {found_model} (matches type: {model_type_preference})")
                                else:
                                    # Явно выбранная модель не соответствует глобальному типу - игнорируем
                                    print(
                                        f"[web] Explicit model for {symbol} "
                                        f"({model_type_from_filename}) doesn't match global preference "
                                        f"({model_type_preference}), ignoring it"
                                    )
                            else:
                                # Глобальный тип не задан - используем явно выбранную модель
                                found_model = str(model_path_obj)
                                print(f"[web] Using explicitly selected model for {symbol}: {found_model}")
            
            # Если явно выбранная модель не найдена или не соответствует символу/типу, ищем автоматически
            if not found_model:
                if model_type_preference:
                    # Ищем модель указанного типа
                    pattern = f"{model_type_preference}_{symbol}_*.pkl"
                    for model_file in sorted(models_dir.glob(pattern), reverse=True):
                        if model_file.is_file():
                            found_model = str(model_file)
                            break
                else:
                    # Автоматический выбор: предпочитаем ensemble > rf > xgb
                    for model_type in ["ensemble", "rf", "xgb"]:
                        pattern = f"{model_type}_{symbol}_*.pkl"
                        for model_file in sorted(models_dir.glob(pattern), reverse=True):
                            if model_file.is_file():
                                found_model = str(model_file)
                                break
                        if found_model:
                            break
            
            if found_model:
                try:
                    metadata = trainer.load_model_metadata(found_model)
                    
                    # Если метаданных нет, извлекаем из имени файла
                    if not metadata:
                        filename = Path(found_model).name
                        parts = filename.replace('.pkl', '').split('_')
                        if len(parts) >= 3:
                            metadata = {
                                "symbol": parts[1],
                                "interval": parts[2],
                                "model_type": parts[0].lower(),
                            }
                    
                    models_info[symbol] = {
                        "model_path": found_model,
                        "model_name": Path(found_model).name,
                        "metadata": metadata or {},
                        "found": True,
                    }
                except Exception as e:
                    print(f"[web] Error loading model info for {symbol}: {e}")
                    models_info[symbol] = {
                        "found": False,
                        "error": str(e),
                    }
            else:
                models_info[symbol] = {
                    "found": False,
                    "error": "Model not found",
                }
        
        return jsonify({
            "models": models_info,
            "model_type_preference": model_type_preference or "auto",
        })
        
    except Exception as e:
        print(f"[web] Error in api_ml_models_all_pairs: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/ml/models/list")
@login_required
def api_ml_models_list():
    """Получить список всех доступных ML моделей."""
    try:
        from pathlib import Path
        # Отложенный импорт для избежания deadlock при многопоточности
        try:
            from bot.ml.model_trainer import ModelTrainer
        except Exception as import_error:
            print(f"[web] Error importing ModelTrainer in api_ml_models_list: {import_error}")
            return jsonify({"error": f"Failed to import ML trainer: {str(import_error)}", "models": []}), 500
        
        models_dir = Path(__file__).parent.parent.parent / "ml_models"
        print(f"[web] Looking for models in: {models_dir}")
        
        if not models_dir.exists():
            print(f"[web] Models directory does not exist: {models_dir}")
            return jsonify({"models": []})
        
        models = []
        trainer = ModelTrainer()
        
        # Ищем все .pkl файлы
        model_files = list(models_dir.glob("*.pkl"))
        for model_file in model_files:
            try:
                # Загружаем метаданные
                metadata = trainer.load_model_metadata(str(model_file))
                
                # Пытаемся извлечь из имени файла, если метаданных нет или в них нет нужных полей
                filename = model_file.name
                parts = filename.replace('.pkl', '').split('_')
                
                # Извлекаем model_type из имени файла (rf, xgb и т.д.)
                model_type_from_filename = None
                symbol_from_filename = None
                interval_from_filename = None
                
                if len(parts) >= 3:
                    model_type_from_filename = parts[0].lower()  # rf или xgb
                    symbol_from_filename = parts[1]
                    interval_from_filename = parts[2]
                
                # Если метаданных нет, создаем их из имени файла
                if not metadata:
                    if model_type_from_filename and symbol_from_filename and interval_from_filename:
                        metadata = {
                            "symbol": symbol_from_filename,
                            "interval": interval_from_filename,
                            "model_type": model_type_from_filename,
                        }
                
                # Если метаданные есть, но в них нет model_type, используем из имени файла
                if metadata and not metadata.get("model_type"):
                    if model_type_from_filename:
                        metadata["model_type"] = model_type_from_filename
                
                # Если метаданные есть, но в них нет symbol или interval, используем из имени файла
                if metadata:
                    if not metadata.get("symbol") and symbol_from_filename:
                        metadata["symbol"] = symbol_from_filename
                    if not metadata.get("interval") and interval_from_filename:
                        metadata["interval"] = interval_from_filename
                
                # Определяем финальные значения
                model_type = "unknown"
                if metadata:
                    model_type = metadata.get("model_type", "unknown")
                    if model_type == "unknown" and model_type_from_filename:
                        model_type = model_type_from_filename
                elif model_type_from_filename:
                    model_type = model_type_from_filename
                
                # Преобразуем model_type в понятное название
                model_type_display = {
                    "rf": "Random Forest",
                    "random_forest": "Random Forest",
                    "xgb": "XGBoost",
                    "xgboost": "XGBoost",
                    "ensemble": "🎯 Ensemble (RF + XGBoost)",
                    "ensemble_weighted": "🎯 Ensemble (Weighted)",
                    "ensemble_voting": "🎯 Ensemble (Voting)",
                    "unknown": "Unknown"
                }.get(model_type.lower(), model_type.upper())
                
                # Получаем дополнительные метрики для ансамбля
                cv_mean = metadata.get("cv_mean", 0.0) if metadata else 0.0
                cv_std = metadata.get("cv_std", 0.0) if metadata else 0.0
                precision = metadata.get("precision", None) if metadata else None
                recall = metadata.get("recall", None) if metadata else None
                f1_score = metadata.get("f1_score", None) if metadata else None
                cv_f1_mean = metadata.get("cv_f1_mean", None) if metadata else None
                
                models.append({
                    "path": str(model_file),
                    "filename": model_file.name,
                    "symbol": metadata.get("symbol", symbol_from_filename) if metadata else (symbol_from_filename or "UNKNOWN"),
                    "interval": metadata.get("interval", interval_from_filename) if metadata else (interval_from_filename or "15"),
                    "model_type": model_type_display,  # Используем понятное название
                    "model_type_raw": model_type,  # Сохраняем исходное значение для совместимости
                    "trained_at": metadata.get("trained_at", None) if metadata else None,
                    "accuracy": metadata.get("accuracy", 0.0) if metadata else 0.0,
                    "cv_mean": cv_mean,
                    "cv_std": cv_std,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "cv_f1_mean": cv_f1_mean,
                    "is_ensemble": "ensemble" in model_type.lower(),
                })
            except Exception as e:
                print(f"[web] Error loading model metadata for {model_file}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Сортируем по дате обучения (новые первыми)
        # Используем безопасную сортировку для случаев, когда trained_at может быть None
        models.sort(key=lambda x: x.get("trained_at") or "", reverse=True)
        
        return jsonify({"models": models})
    except Exception as e:
        print(f"[web] Error in api_ml_models_list: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/ml/model/select", methods=["POST"])
@login_required
def api_ml_model_select():
    """Выбрать активную ML модель."""
    if not settings:
        return jsonify({"error": "Settings not loaded"}), 500
    
    data = request.json or {}
    model_path = data.get("model_path")
    
    if not model_path:
        return jsonify({"error": "model_path is required"}), 400
    
    from pathlib import Path
    model_file = Path(model_path)
    
    if not model_file.exists():
        return jsonify({"error": f"Model file not found: {model_path}"}), 404
    
    try:
        # Обновляем настройки
        settings.ml_model_path = str(model_path)
        
        # Сохраняем в .env
        _save_settings_to_env(settings)
        
        # Обновляем в shared_settings
        from bot.shared_settings import set_settings
        set_settings(settings)
        
        return jsonify({
            "success": True,
            "message": f"Model selected: {model_file.name}",
            "model_path": str(model_path),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ml/model/retrain", methods=["POST"])
@login_required
def api_ml_model_retrain():
    """Переобучить ML модель для одной пары."""
    if not settings:
        return jsonify({"error": "Settings not loaded"}), 500
    
    try:
        data = request.json or {}
        symbol = data.get("symbol", settings.symbol)
        mode = data.get("mode", "optimal")
        
        # Проверяем доступные пары
        available_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        if symbol not in available_symbols:
            return jsonify({"error": f"Symbol {symbol} not supported. Available: {available_symbols}"}), 400

        script_name = "retrain_ml_optimized.py" if mode == "optimal" else "retrain_ultra_aggressive.py"
        mode_display = "Оптимальный" if mode == "optimal" else "Агрессивный"
        
        def run_single_retrain():
            try:
                import subprocess
                import sys
                print(f"[web] 🚀 Запуск {mode_display} переобучения для {symbol}...")
                subprocess.run([sys.executable, script_name, "--symbol", symbol], check=True)
                print(f"[web] ✅ {mode_display} переобучение {symbol} завершено!")
            except Exception as e:
                print(f"[web] ❌ Ошибка {symbol} ({mode_display}): {e}")

        import threading
        thread = threading.Thread(target=run_single_retrain, daemon=True)
        thread.start()
        
        return jsonify({
            "success": True,
            "message": f"Запущено {mode_display} переобучение для {symbol}",
            "status": "training",
            "symbol": symbol,
            "mode": mode
        })
        
    except Exception as e:
        return jsonify({"error": f"Не удалось запустить обучение: {e}"}), 500


@app.route("/api/ml/model/retrain-all", methods=["POST"])
@login_required
def api_ml_model_retrain_all():
    """Переобучить ML модели для всех пар (BTCUSDT, ETHUSDT, SOLUSDT)."""
    if not settings:
        return jsonify({"error": "Settings not loaded"}), 500
    
    try:
        data = request.json or {}
        mode = data.get("mode", "optimal")  # 'optimal' или 'aggressive'
        
        script_name = "retrain_ml_optimized.py" if mode == "optimal" else "retrain_ultra_aggressive.py"
        mode_display = "Оптимальный" if mode == "optimal" else "Агрессивный"
        
        # Проверяем, не запущен ли уже какой-либо процесс обучения
        import subprocess
        import sys
        
        # Определяем, какой процесс искать
        current_script = os.path.basename(script_name)

        def run_retrain_script():
            try:
                print(f"[web] 🚀 Запуск {mode_display} переобучения всех моделей...")
                import sys
                import os
                
                # Запускаем скрипт через текущий интерпретатор (venv)
                # Передаем --mode если скрипт это поддерживает, или просто запускаем нужный файл
                subprocess.run([sys.executable, script_name], check=True)
                print(f"[web] ✅ {mode_display} переобучение завершено!")
                
            except Exception as e:
                print(f"[web] ❌ Ошибка при выполнении {script_name}: {e}")

        # Запускаем в отдельном потоке
        import threading
        thread = threading.Thread(target=run_retrain_script, daemon=True)
        thread.start()
        
        return jsonify({
            "success": True,
            "message": f"Запущено {mode_display} переобучение для всех пар (BTCUSDT, ETHUSDT, SOLUSDT)",
            "status": "training",
            "mode": mode
        })
        
    except Exception as e:
        return jsonify({"error": f"Не удалось запустить обучение: {e}"}), 500


@app.route("/api/signals")
@login_required
def api_signals():
    """Получить историю сигналов."""
    # Получаем символ из query параметра или используем primary_symbol
    symbol_filter = request.args.get("symbol", None)
    if not symbol_filter and settings:
        symbol_filter = settings.primary_symbol if settings.primary_symbol else settings.symbol
    
    signals = get_signals(limit=100, symbol_filter=symbol_filter)
    return jsonify({
        "signals": signals,
        "total": len(signals),
        "symbol": symbol_filter or (settings.primary_symbol if settings else "ALL"),
    })


@app.route("/api/signals/clear", methods=["POST"])
@login_required
def api_clear_signals():
    """Очистить всю историю сигналов."""
    try:
        from bot.web.history import clear_signals
        clear_signals()
        return jsonify({
            "success": True,
            "message": "All signals cleared",
        })
    except Exception as e:
        return jsonify({"error": f"Failed to clear signals: {e}"}), 500


@app.route("/api/wallet")
@login_required
def api_wallet():
    """Получить информацию о кошельке и балансе."""
    if not client or not settings:
        return jsonify({"error": "Client or settings not initialized"}), 500
    
    try:
        # Получаем символ из query параметра или используем primary_symbol
        symbol_from_query = request.args.get("symbol", None)
        if symbol_from_query and symbol_from_query in (settings.active_symbols if settings.active_symbols else ["BTCUSDT", "ETHUSDT", "SOLUSDT"]):
            selected_symbol = symbol_from_query
        else:
            selected_symbol = settings.primary_symbol if settings.primary_symbol else (settings.symbol if settings.symbol else "BTCUSDT")
        
        # Получаем баланс кошелька
        wallet_resp = client.get_wallet_balance(account_type="UNIFIED")
        
        # Получаем информацию о позициях
        # ВАЖНО: cumRealisedPnl из этого запроса используется для блока "Trading"
        position_resp = client.get_position_info(symbol=selected_symbol)
        
        # Получаем открытые ордера
        orders_resp = client.get_open_orders(symbol=selected_symbol)
        
        # Парсим данные о балансе
        wallet_data = {}
        
        def safe_float(value, default=0.0):
            """Безопасное преобразование в float, обрабатывает пустые строки и None."""
            if value is None or value == "":
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        
        if wallet_resp.get("retCode") == 0:
            result = wallet_resp.get("result", {})
            list_data = result.get("list", [])
            if list_data:
                account = list_data[0]  # Берем первый аккаунт
                coin_list = account.get("coin", [])
                
                # Ищем USDT - наша основная монета
                usdt_coin = None
                for coin in coin_list:
                    if coin.get("coin") == "USDT":
                        usdt_coin = coin
                        break
                
                if usdt_coin:
                    # Маржа в позиции = totalPositionIM
                    # Общий реализованный PnL = cumRealisedPnl
                    wallet_balance = safe_float(usdt_coin.get("walletBalance", 0))
                    used_margin = safe_float(usdt_coin.get("totalPositionIM", 0))
                    # Доступный баланс = Wallet Balance - Position Margin (свободный баланс для новых позиций)
                    available_balance = wallet_balance - used_margin
                    
                    wallet_data = {
                        "available_balance": available_balance,  # Доступный баланс = Wallet Balance - Position Margin
                        "wallet_balance": wallet_balance,
                        "used_margin": used_margin,  # Маржа в позиции = totalPositionIM
                        "bonus": safe_float(usdt_coin.get("bonus", 0)),
                        "accrued_interest": safe_float(usdt_coin.get("accruedInterest", 0)),
                        "cum_realised_pnl": safe_float(usdt_coin.get("cumRealisedPnl", 0)),  # Общий реализованный PnL
                    }
        
        # Парсим данные о позициях
        position_data = None
        unrealised_pnl = 0.0
        cum_realised_pnl_from_position = None  # cumRealisedPnl из позиции (архивные данные), None означает "не найдено"
        
        if position_resp.get("retCode") == 0:
            result = position_resp.get("result", {})
            list_data = result.get("list", [])
            if list_data:
                # Проходим по всем позициям, чтобы найти cumRealisedPnl (архивные данные)
                # cumRealisedPnl может быть даже если позиция закрыта (size == 0)
                for pos in list_data:
                    # Извлекаем cumRealisedPnl из всех позиций (даже закрытых)
                    cum_realised_pnl_raw = pos.get("cumRealisedPnl")
                    if cum_realised_pnl_raw is None or cum_realised_pnl_raw == "":
                        cum_realised_pnl_raw = pos.get("cum_realised_pnl")
                    
                    if cum_realised_pnl_raw is not None and cum_realised_pnl_raw != "":
                        cum_realised_pnl_value = safe_float(cum_realised_pnl_raw, 0)
                        # Берем последнее значение (даже если 0 или отрицательное) - это архивные данные
                        # Если есть несколько позиций, берем последнюю
                        cum_realised_pnl_from_position = cum_realised_pnl_value
                    
                    # Для открытой позиции (size != 0) сохраняем полные данные
                    size = safe_float(pos.get("size", 0))
                    if size != 0 and position_data is None:
                        position_data = {
                            "side": pos.get("side", ""),
                            "size": size,
                            "avg_price": safe_float(pos.get("avgPrice", 0)),
                            "mark_price": safe_float(pos.get("markPrice", 0)),
                            "unrealised_pnl": safe_float(pos.get("unrealisedPnl", 0)),
                            "leverage": pos.get("leverage", "1"),
                            "take_profit": pos.get("takeProfit", ""),
                            "stop_loss": pos.get("stopLoss", ""),
                            "cum_realised_pnl": safe_float(cum_realised_pnl_raw, 0) if cum_realised_pnl_raw else 0,  # cumRealisedPnl из позиции
                        }
                        unrealised_pnl = position_data["unrealised_pnl"]
        
        # Парсим данные об ордерах
        open_orders_count = 0
        orders_margin = 0.0
        tp_orders = []  # Список TP ордеров
        sl_orders = []  # Список SL ордеров
        if orders_resp.get("retCode") == 0:
            result = orders_resp.get("result", {})
            list_data = result.get("list", [])
            open_orders_count = len(list_data) if list_data else 0
            # Рассчитываем примерную маржу для открытых ордеров и собираем TP/SL ордера
            for order in list_data:
                qty = safe_float(order.get("qty", 0))
                price = safe_float(order.get("price", 0))
                if qty > 0 and price > 0:
                    orders_margin += (qty * price) / settings.leverage
                
                # Проверяем, является ли ордер TP или SL
                order_type = order.get("orderType", "").upper()
                stop_order_type = order.get("stopOrderType", "").upper()
                trigger_price = safe_float(order.get("triggerPrice", 0))
                take_profit = safe_float(order.get("takeProfit", 0))
                stop_loss = safe_float(order.get("stopLoss", 0))
                
                # Рассчитываем маржу для ордера
                order_margin = 0.0
                if qty > 0:
                    order_price = price if price > 0 else (trigger_price if trigger_price > 0 else 0)
                    if order_price > 0:
                        order_margin = (qty * order_price) / settings.leverage
                
                # TP ордера: TakeProfit или ордера с triggerPrice выше текущей цены для LONG
                if take_profit > 0 or (stop_order_type == "TAKEPROFIT"):
                    tp_orders.append({
                        "order_id": order.get("orderId", ""),
                        "symbol": order.get("symbol", ""),
                        "side": order.get("side", ""),
                        "qty": qty,
                        "trigger_price": trigger_price if trigger_price > 0 else take_profit,
                        "price": price if price > 0 else take_profit,
                        "margin": order_margin,
                        "status": order.get("orderStatus", ""),
                    })
                
                # SL ордера: StopLoss или ордера с triggerPrice ниже текущей цены для LONG
                if stop_loss > 0 or (stop_order_type == "STOPLOSS"):
                    sl_orders.append({
                        "order_id": order.get("orderId", ""),
                        "symbol": order.get("symbol", ""),
                        "side": order.get("side", ""),
                        "qty": qty,
                        "trigger_price": trigger_price if trigger_price > 0 else stop_loss,
                        "price": price if price > 0 else stop_loss,
                        "margin": order_margin,
                        "status": order.get("orderStatus", ""),
                    })
        
        # Общий реализованный PnL для блока "Trading" всегда берем из cumRealisedPnl позиции (архивные данные)
        # cumRealisedPnl из запроса позиции (/v5/position/list) - это архивные данные по позиции
        # НЕ путать с cumRealisedPnl из кошелька, который используется в первом блоке "Wallet Balance"
        if cum_realised_pnl_from_position is not None:
            # Если мы нашли cumRealisedPnl в позиции (даже если 0 или отрицательное), используем его
            total_pnl = cum_realised_pnl_from_position
        else:
            # Fallback на кошелек только если cumRealisedPnl не найден в позиции
            total_pnl = wallet_data.get("cum_realised_pnl", 0)
        
        return jsonify({
            "wallet": wallet_data,
            "position": position_data,  # position_data уже содержит cum_realised_pnl
            "open_orders": {
                "count": open_orders_count,
                "margin_used": orders_margin,
                "tp_orders": tp_orders,
                "sl_orders": sl_orders,
                "tp_count": len(tp_orders),
                "sl_count": len(sl_orders),
            },
            "trading": {
                "unrealised_pnl": unrealised_pnl,
                "total_pnl": total_pnl,  # Это значение из позиции или кошелька
            },
            "symbol": selected_symbol,  # Возвращаем выбранный символ
        })
    except Exception as e:
        print(f"[web] Error getting wallet info: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/bot/start", methods=["POST"])
@login_required
def api_bot_start():
    """Запустить бота для всех активных символов."""
    global bot_thread, bot_stop_event, multi_symbol_manager, settings, bot_state, client
    
    try:
        # Дополнительная проверка инициализации перед запуском бота
        if settings is None or client is None:
            print("[web] Error: Application not initialized")
            try:
                init_app()
                print("[web] Application initialized successfully")
            except Exception as e:
                print(f"[web] Failed to initialize application: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    "success": False,
                    "error": f"Failed to initialize application: {str(e)}. Please check server logs."
                }), 500
        
        if not settings:
            print("[web] Error: Settings not loaded")
            return jsonify({"success": False, "error": "Settings not loaded. Please check .env file."}), 500
        
        # Проверяем, используется ли MultiSymbolManager
        if multi_symbol_manager is None:
            # Инициализируем MultiSymbolManager, если он не был инициализирован
            print("[web] Warning: MultiSymbolManager not initialized, initializing now...")
            try:
                from bot.multi_symbol_manager import MultiSymbolManager
                multi_symbol_manager = MultiSymbolManager(settings)
                print(f"[web] MultiSymbolManager initialized with active symbols: {settings.active_symbols}")
            except Exception as e:
                print(f"[web] Error initializing MultiSymbolManager: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({"success": False, "error": f"Failed to initialize MultiSymbolManager: {str(e)}"}), 500
        
        # Проверяем наличие активных символов
        if not settings.active_symbols or len(settings.active_symbols) == 0:
            print("[web] Warning: No active symbols configured in settings")
            print(f"[web] Settings.active_symbols = {settings.active_symbols}")
            print(f"[web] Settings.symbol = {getattr(settings, 'symbol', 'Not set')}")
            
            # Fallback на старый режим с одним символом
            if not hasattr(settings, 'symbol') or not settings.symbol:
                print("[web] Error: No active symbols and no default symbol configured")
                return jsonify({
                    "success": False, 
                    "error": "No active symbols configured. Please go to the 'Symbols' tab and select at least one trading pair.",
                    "details": "ACTIVE_SYMBOLS is empty in .env file"
                }), 400
            else:
                # Используем единственный символ как активный
                print(f"[web] Using fallback single symbol mode with symbol: {settings.symbol}")
                settings.active_symbols = [settings.symbol]
        
        print(f"[web] Starting bot with active symbols: {settings.active_symbols}")
        print(f"[web] Active symbols count: {len(settings.active_symbols) if settings.active_symbols else 0}")
        print(f"[web] MultiSymbolManager status: running={multi_symbol_manager.running if multi_symbol_manager else 'None'}")
        print(f"[web] MultiSymbolManager instance: {multi_symbol_manager}")
        
        # Используем MultiSymbolManager для многопарной торговли
        if multi_symbol_manager and settings.active_symbols and len(settings.active_symbols) > 0:
            print(f"[web] Using MultiSymbolManager for {len(settings.active_symbols)} symbol(s)")
            if multi_symbol_manager.running:
                print("[web] Warning: MultiSymbolManager is already running")
                return jsonify({"success": False, "error": "Bot is already running"}), 400
            
            try:
                # Обновляем bot_state ПЕРЕД запуском, чтобы UI мог сразу увидеть изменения
                bot_state["is_running"] = True
                bot_state["current_status"] = "Starting"
                bot_state["last_update"] = datetime.now(timezone.utc).isoformat()
                
                # Запускаем обновление настроек и запуск менеджера в отдельном потоке, чтобы не блокировать Flask
                print("[web] ⚙️  Starting bot in background thread (update_settings + start)...")
                print("[web] ⏳ Bot is starting in background (this won't block the server)...")
                
                def start_bot_background():
                    """Запуск бота в фоновом потоке (обновление настроек + запуск)."""
                    global bot_state, settings, multi_symbol_manager
                    
                    try:
                        # Шаг 1: Обновляем настройки менеджера
                        print("[web] [background] Step 1: Updating MultiSymbolManager settings...")
                        try:
                            multi_symbol_manager.update_settings(settings)
                            print("[web] [background] ✅ Step 1 complete: MultiSymbolManager settings updated successfully")
                        except Exception as e:
                            error_msg = str(e)
                            print(f"[web] [background] ❌ ERROR in Step 1 (update_settings): {error_msg}")
                            import traceback
                            error_trace = traceback.format_exc()
                            print(f"[web] [background] Full traceback:\n{error_trace}")
                            raise
                        
                        # Шаг 2: Запускаем менеджер
                        print("[web] [background] Step 2: Starting MultiSymbolManager.start()...")
                        multi_symbol_manager.start()
                        print("[web] [background] ✅ Step 2 complete: MultiSymbolManager.start() completed successfully")
                        
                        # Даем немного времени воркерам на инициализацию
                        import time
                        time.sleep(0.5)  # Короткая задержка для проверки статуса
                        
                        # Проверяем статус воркеров после запуска
                        all_workers_status = multi_symbol_manager.get_all_workers_status()
                        active_symbols = settings.active_symbols if settings and hasattr(settings, 'active_symbols') else []
                        alive_workers = []
                        
                        for symbol in active_symbols:
                            worker = multi_symbol_manager.workers.get(symbol) if hasattr(multi_symbol_manager, 'workers') else None
                            if worker and hasattr(worker, 'thread') and worker.thread and worker.thread.is_alive():
                                alive_workers.append(symbol)
                                print(f"[web] [background] ✅ Worker for {symbol} is ALIVE (thread ID: {worker.thread.ident})")
                            else:
                                print(f"[web] [background] ⚠️ Worker for {symbol} is NOT ALIVE")
                        
                        if alive_workers:
                            bot_state["is_running"] = True
                            bot_state["current_status"] = "Running"
                            bot_state["last_action"] = f"Started for {', '.join(alive_workers)}"
                            bot_state["last_action_time"] = datetime.now(timezone.utc).isoformat()
                            bot_state["last_update"] = datetime.now(timezone.utc).isoformat()
                            print(f"[web] [background] 🎉 Bot started successfully for symbols: {', '.join(alive_workers)}")
                        else:
                            print("[web] [background] ⚠️ Warning: No alive workers found after startup")
                            bot_state["is_running"] = False
                            bot_state["current_status"] = "Error"
                            bot_state["last_error"] = "No alive workers found after startup"
                            bot_state["last_error_time"] = datetime.now(timezone.utc).isoformat()
                    except Exception as e:
                        error_msg = str(e)
                        print(f"[web] [background] ❌ ERROR: multi_symbol_manager.start() failed: {error_msg}")
                        import traceback
                        error_trace = traceback.format_exc()
                        print(f"[web] [background] Full traceback:\n{error_trace}")
                        
                        bot_state["is_running"] = False
                        bot_state["current_status"] = "Error"
                        bot_state["last_error"] = f"Failed to start MultiSymbolManager: {error_msg}"
                        bot_state["last_error_time"] = datetime.now(timezone.utc).isoformat()
                
                # Запускаем в отдельном потоке, чтобы не блокировать Flask
                import threading
                start_thread = threading.Thread(
                    target=start_bot_background,
                    name="BotStartThread",
                    daemon=True
                )
                start_thread.start()
                print("[web] ✅ Bot start thread launched - server remains responsive!")
                
                # НЕ ЖДЕМ завершения - сразу возвращаем ответ, чтобы не блокировать Flask
                # Статус будет обновляться в фоновом потоке
                return jsonify({
                    "success": True, 
                    "message": f"Bot is starting in background for symbols: {', '.join(settings.active_symbols)}. Status will update shortly.",
                    "active_symbols": settings.active_symbols,
                    "bot_status": {
                        "is_running": True,  # Устанавливаем как "starting"
                        "current_status": "Starting",
                        "last_update": bot_state["last_update"]
                    }
                })
            except Exception as e:
                error_msg = str(e)
                print(f"[web] Error starting MultiSymbolManager: {error_msg}")
                import traceback
                error_trace = traceback.format_exc()
                print(f"[web] Full traceback:\n{error_trace}")
                
                bot_state["is_running"] = False
                bot_state["current_status"] = "Error"
                bot_state["last_error"] = error_msg
                bot_state["last_error_time"] = datetime.now(timezone.utc).isoformat()
                
                # Возвращаем детальную информацию об ошибке для отладки
                return jsonify({
                    "success": False, 
                    "error": f"Failed to start bot: {error_msg}",
                    "details": error_trace.split('\n')[-5:] if len(error_trace.split('\n')) > 5 else [error_trace]
                }), 500
        else:
            # Обратная совместимость: запускаем старый способ (один символ)
            if bot_state["is_running"]:
                return jsonify({"success": False, "error": "Bot is already running"}), 400
            
            # Сбрасываем событие остановки
            bot_stop_event.clear()
            
            # Запускаем бота в отдельном потоке
            try:
                bot_thread = threading.Thread(
                    target=_run_bot_in_thread,
                    args=(settings,),
                    daemon=True,
                    name="BotThread"
                )
                bot_thread.start()
                
                bot_state["is_running"] = True
                bot_state["last_update"] = datetime.now(timezone.utc).isoformat()
                return jsonify({"success": True, "message": f"Bot started for {settings.symbol}"})
            except Exception as e:
                print(f"[web] Error starting bot thread: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({"success": False, "error": f"Failed to start bot thread: {str(e)}"}), 500
    
    except Exception as e:
        print(f"[web] Unexpected error in api_bot_start: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Unexpected error: {str(e)}"}), 500


@app.route("/api/position/close", methods=["POST"])
@login_required
def api_position_close():
    """Принудительно закрыть позицию для указанного символа."""
    global client, settings
    
    try:
        ensure_initialized()
        
        # Получаем символ из query параметра
        symbol = request.args.get("symbol", None)
        if not symbol:
            symbol = settings.primary_symbol if settings and settings.primary_symbol else (settings.symbol if settings else "BTCUSDT")
        
        # Получаем информацию о позиции
        position = _get_position(client, symbol)
        
        if not position or not position.get("side") or float(position.get("size", 0)) <= 0:
            return jsonify({
                "success": False,
                "error": f"No open position found for {symbol}"
            }), 400
        
        # Определяем сторону для закрытия (противоположная стороне позиции)
        position_side = position.get("side", "").upper()
        if position_side == "LONG" or position_side == "BUY":
            close_side = "Sell"
        elif position_side == "SHORT" or position_side == "SELL":
            close_side = "Buy"
        else:
            return jsonify({
                "success": False,
                "error": f"Unknown position side: {position_side}"
            }), 400
        
        # Получаем размер позиции
        position_size = float(position.get("size", 0))
        
        if position_size <= 0:
            return jsonify({
                "success": False,
                "error": f"Invalid position size: {position_size}"
            }), 400
        
        # Закрываем позицию через Market ордер с reduce_only=True
        from bot.exchange.bybit_client import BybitClient
        from bot.config import load_settings
        
        current_settings = load_settings()
        bybit_client = BybitClient(current_settings.api)
        
        resp = bybit_client.place_order(
            symbol=symbol,
            side=close_side,
            qty=position_size,
            order_type="Market",
            reduce_only=True,
        )
        
        if resp.get("retCode") == 0:
            print(f"[web] Position closed successfully for {symbol}: {position_size} @ {close_side}")
            return jsonify({
                "success": True,
                "message": f"Position closed successfully for {symbol}",
                "symbol": symbol,
                "side": position_side,
                "size": position_size,
            })
        else:
            error_msg = resp.get("retMsg", "Unknown error")
            error_code = resp.get("retCode", "Unknown")
            print(f"[web] Failed to close position for {symbol}: {error_msg} (ErrCode: {error_code})")
            return jsonify({
                "success": False,
                "error": f"Failed to close position: {error_msg}",
                "error_code": error_code
            }), 400
            
    except Exception as e:
        print(f"[web] Error closing position: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to close position: {str(e)}"}), 500


@app.route("/api/bot/stop", methods=["POST"])
@login_required
def api_bot_stop():
    """Остановить бота для всех символов."""
    global bot_thread, bot_stop_event, multi_symbol_manager, bot_state
    
    try:
        if not bot_state["is_running"]:
            print("[web] Warning: Bot is not running, but stop was called")
            return jsonify({"success": False, "error": "Bot is not running"}), 400
        
        # Проверяем, используется ли MultiSymbolManager
        if multi_symbol_manager and multi_symbol_manager.running:
            # Останавливаем MultiSymbolManager
            try:
                print("[web] Stopping MultiSymbolManager...")
                multi_symbol_manager.stop()
                print("[web] MultiSymbolManager stopped successfully")
                
                bot_state["is_running"] = False
                bot_state["current_status"] = "Stopped"
                bot_state["last_action"] = "Stopped all workers"
                bot_state["last_action_time"] = datetime.now(timezone.utc).isoformat()
                bot_state["last_update"] = datetime.now(timezone.utc).isoformat()
                
                print(f"[web] Bot state updated: is_running={bot_state['is_running']}, status={bot_state['current_status']}")
                
                return jsonify({
                    "success": True, 
                    "message": "Bot stopped for all symbols",
                    "bot_status": {
                        "is_running": bot_state["is_running"],
                        "current_status": bot_state["current_status"],
                        "last_update": bot_state["last_update"]
                    }
                })
            except Exception as e:
                print(f"[web] Error stopping MultiSymbolManager: {e}")
                import traceback
                traceback.print_exc()
                bot_state["is_running"] = False
                bot_state["current_status"] = "Error"
                bot_state["last_error"] = str(e)
                bot_state["last_error_time"] = datetime.now(timezone.utc).isoformat()
                return jsonify({
                    "success": False, 
                    "error": f"Failed to stop bot: {str(e)}"
                }), 500
        else:
            # Обратная совместимость: останавливаем старым способом
            print("[web] Stopping bot (single symbol mode)...")
            # Устанавливаем событие остановки
            bot_stop_event.set()
            
            # Ждем завершения потока (максимум 5 секунд)
            if bot_thread and bot_thread.is_alive():
                bot_thread.join(timeout=5.0)
            
            bot_state["is_running"] = False
            bot_state["current_status"] = "Stopped"
            bot_state["last_action"] = "Stopped"
            bot_state["last_action_time"] = datetime.now(timezone.utc).isoformat()
            bot_state["last_update"] = datetime.now(timezone.utc).isoformat()
            bot_thread = None
            
            print(f"[web] Bot stopped (single symbol mode)")
            return jsonify({
                "success": True, 
                "message": "Bot stopped",
                "bot_status": {
                    "is_running": bot_state["is_running"],
                    "current_status": bot_state["current_status"],
                    "last_update": bot_state["last_update"]
                }
            })
    
    except Exception as e:
        print(f"[web] Unexpected error in api_bot_stop: {e}")
        import traceback
        traceback.print_exc()
        bot_state["is_running"] = False
        bot_state["current_status"] = "Error"
        bot_state["last_error"] = str(e)
        bot_state["last_error_time"] = datetime.now(timezone.utc).isoformat()
        return jsonify({
            "success": False, 
            "error": f"Unexpected error: {str(e)}"
        }), 500


def _run_bot_in_thread(settings: AppSettings):
    """Запускает бота в отдельном потоке."""
    try:
        print(f"[web] Bot thread started")
        bot_state["current_status"] = "Starting"
        bot_state["last_update"] = datetime.now(timezone.utc).isoformat()
        # Запускаем live бота с передачей bot_state для обновления статуса
        run_live_from_api(settings, bot_state=bot_state)
    except Exception as e:
        print(f"[web] Error in bot thread: {e}")
        import traceback
        traceback.print_exc()
        bot_state["is_running"] = False
        bot_state["current_status"] = "Error"
        bot_state["last_error"] = str(e)
        bot_state["last_error_time"] = datetime.now(timezone.utc).isoformat()
    finally:
        print(f"[web] Bot thread finished")
        bot_state["is_running"] = False
        bot_state["current_status"] = "Stopped"


@app.route("/api/chart/data")
@login_required
def api_chart_data():
    """Получить данные для графика: свечи, индикаторы и сигналы."""
    if not client or not settings:
        return jsonify({"error": "Client or settings not initialized"}), 500
    
    try:
        # Получаем символ из query параметра или используем primary_symbol
        symbol = request.args.get('symbol', None)
        if not symbol:
            symbol = settings.primary_symbol if settings.primary_symbol else settings.symbol
        
        # Валидация символа
        available_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        if symbol not in available_symbols:
            symbol = settings.primary_symbol if settings.primary_symbol else settings.symbol
        
        # Конвертируем время в МСК (UTC+3) - определяем в начале функции
        msk_tz = pytz.timezone('Europe/Moscow')
        utc_tz = pytz.UTC
        
        # Получаем текущую позицию для выбранного символа
        position = _get_position(client, symbol)
        
        # Для графика всегда используем 15m таймфрейм за последние 3 дня
        # 3 дня * 24 часа * 4 свечи в час (15m) = 288 свечей
        chart_timeframe = "15m"
        chart_interval = _timeframe_to_bybit_interval(chart_timeframe)
        chart_limit = 288  # 3 дня на 15m
        
        df_raw = client.get_kline_df(symbol=symbol, interval=chart_interval, limit=chart_limit)
        
        if df_raw.empty:
            return jsonify({"error": "No raw data received from exchange"}), 500
        
        _web_log(f"[web] Raw data: {len(df_raw)} candles for {symbol}")
        
        # Вычисляем индикаторы
        try:
            df_ind = prepare_with_indicators(
                df_raw,
                adx_length=settings.strategy.adx_length,
                di_length=settings.strategy.di_length,
                sma_length=settings.strategy.sma_length,
                rsi_length=settings.strategy.rsi_length,
                breakout_lookback=settings.strategy.breakout_lookback,
                bb_length=settings.strategy.bb_length,
                bb_std=settings.strategy.bb_std,
                atr_length=14,  # ATR период
                ema_fast_length=settings.strategy.ema_fast_length,
                ema_slow_length=settings.strategy.ema_slow_length,
                ema_timeframe=settings.strategy.momentum_ema_timeframe,
            )
            print(f"[web] After prepare_with_indicators: {len(df_ind)} candles")
        except Exception as e:
            print(f"[web] Error in prepare_with_indicators: {e}")
            import traceback
            print(f"[web] Traceback: {traceback.format_exc()}")
            return jsonify({"error": f"Error calculating indicators: {str(e)}"}), 500
        
        if df_ind.empty:
            # Если после вычисления индикаторов DataFrame пуст, используем исходные данные
            print(f"[web] ⚠️ Warning: df_ind is empty after prepare_with_indicators, using raw data")
            df_ind = df_raw.copy()
            # Добавляем минимальные индикаторы вручную
            try:
                import pandas_ta as ta
                df_ind["sma"] = ta.sma(df_ind["close"], length=settings.strategy.sma_length)
                df_ind["rsi"] = ta.rsi(df_ind["close"], length=settings.strategy.rsi_length)
                df_ind["adx"] = 0  # Заглушка
                df_ind["bb_upper"] = df_ind["close"] * 1.02
                df_ind["bb_middle"] = df_ind["close"]
                df_ind["bb_lower"] = df_ind["close"] * 0.98
            except Exception as e:
                print(f"[web] Error adding minimal indicators: {e}")
        
        df_ready = enrich_for_strategy(df_ind, settings.strategy)
        print(f"[web] After enrich_for_strategy: {len(df_ready)} candles")
        
        if df_ready.empty:
            # Если после enrich_for_strategy DataFrame пуст, используем df_ind
            print(f"[web] ⚠️ Warning: df_ready is empty after enrich_for_strategy, using df_ind")
            df_ready = df_ind.copy()
        
        # Загружаем сигналы из истории для отображения на графике
        # Это позволяет показать исторические сигналы, даже если стратегия сейчас выключена
        historical_signals = []
        try:
            from bot.web.history import get_signals
            from bot.strategy import Signal
            
            # Получаем последние сигналы из истории
            all_historical = get_signals(limit=500)  # Берем больше, чтобы покрыть 3 дня
            
            # Фильтруем по символу и временному диапазону графика (последние 3 дня)
            chart_start_time = None
            if df_ready is not None and not df_ready.empty:
                try:
                    # Первое время на графике (индекс первого бара)
                    first_idx = df_ready.index[0]
                    if isinstance(first_idx, pd.Timestamp):
                        chart_start_time = first_idx
                        if chart_start_time.tzinfo is None:
                            chart_start_time = utc_tz.localize(chart_start_time)
                        else:
                            chart_start_time = chart_start_time.astimezone(utc_tz)
                except Exception as e:
                    print(f"[web] Error getting chart start time: {e}")
            
            for hist_sig in all_historical:
                # Проверяем символ
                if hist_sig.get("symbol", "").upper() != symbol.upper():
                    continue
                
                # Проверяем действие (только LONG и SHORT)
                action_str = hist_sig.get("action", "").lower()
                if action_str not in ("long", "short"):
                    continue
                
                # Парсим время сигнала
                sig_time_str = hist_sig.get("timestamp", "")
                if not sig_time_str:
                    continue
                
                try:
                    # Парсим время сигнала
                    if isinstance(sig_time_str, str):
                        if 'T' in sig_time_str:
                            sig_time = pd.to_datetime(sig_time_str.replace('Z', '+00:00'))
                        else:
                            sig_time = pd.to_datetime(sig_time_str)
                    else:
                        sig_time = pd.to_datetime(sig_time_str)
                    
                    if sig_time.tzinfo is None:
                        sig_time = utc_tz.localize(sig_time)
                    else:
                        sig_time = sig_time.astimezone(utc_tz)
                    
                    # Проверяем, попадает ли сигнал во временной диапазон графика
                    if chart_start_time:
                        # Добавляем небольшой буфер (1 час) для учета возможных расхождений
                        if sig_time < (chart_start_time - timedelta(hours=1)):
                            continue
                    
                    # Преобразуем action в Action enum
                    action_enum = Action.LONG if action_str == "long" else Action.SHORT
                    
                    # Создаем объект Signal
                    hist_sig_obj = Signal(
                        timestamp=sig_time,
                        action=action_enum,
                        reason=hist_sig.get("reason", ""),
                        price=float(hist_sig.get("price", 0.0)),
                    )
                    historical_signals.append(hist_sig_obj)
                except Exception as e:
                    print(f"[web] Error converting historical signal to Signal object: {e}, signal: {hist_sig}")
                    continue
        except Exception as e:
            print(f"[web] Error loading historical signals: {e}")
            import traceback
            print(f"[web] Traceback: {traceback.format_exc()}")
            historical_signals = []
        
        # Загружаем ВСЕ сигналы: исторические (из bot_history.json) + генерируемые сейчас (если стратегии включены)
        # Исторические сигналы отображаются всегда, независимо от настроек стратегий
        signals = []
        # Добавляем исторические сигналы первыми (это сигналы от всех стратегий, сохраненные ранее)
        signals.extend(historical_signals)
        
        # Также генерируем текущие сигналы от включенных стратегий (для отображения в реальном времени)
        try:
            # Trend стратегия - генерируем текущие сигналы (если стратегия включена)
            if settings.enable_trend_strategy:
                use_momentum = settings.enable_momentum_strategy
                trend_signals = build_signals(df_ready, settings.strategy, use_momentum=use_momentum, use_liquidity=False)
                _web_log(f"[web] Strategy processed")
                for sig in trend_signals:
                    # Добавляем только LONG и SHORT сигналы (HOLD не показываем)
                    if sig.reason.startswith("trend_") and sig.action in (Action.LONG, Action.SHORT):
                        signals.append(sig)
                        # Сохраняем сигнал в историю для синхронизации с графиком
                        try:
                            from bot.web.history import add_signal
                            ts_log = sig.timestamp
                            if isinstance(ts_log, pd.Timestamp):
                                if ts_log.tzinfo is None:
                                    ts_log = ts_log.tz_localize('UTC')
                                else:
                                    ts_log = ts_log.tz_convert('UTC')
                                ts_log = ts_log.to_pydatetime()
                            add_signal(
                                action=sig.action.value,
                                reason=sig.reason,
                                price=sig.price,
                                timestamp=ts_log,
                                symbol=symbol,
                                strategy_type="trend",
                                signal_id=sig.signal_id if hasattr(sig, 'signal_id') and sig.signal_id else None,
                            )
                        except Exception as e:
                            print(f"[web] ⚠️ Failed to save trend signal to history: {e}")
            
            # Flat стратегия - генерируем текущие сигналы (если стратегия включена)
            if settings.enable_flat_strategy:
                flat_signals = build_signals(df_ready, settings.strategy, use_momentum=False, use_liquidity=False)
                _web_log(f"[web] Strategy processed")
                for sig in flat_signals:
                    # Добавляем только LONG и SHORT сигналы (HOLD не показываем)
                    if sig.reason.startswith("range_") and sig.action in (Action.LONG, Action.SHORT):
                        signals.append(sig)
                        # Сохраняем сигнал в историю для синхронизации с графиком
                        try:
                            from bot.web.history import add_signal
                            ts_log = sig.timestamp
                            if isinstance(ts_log, pd.Timestamp):
                                if ts_log.tzinfo is None:
                                    ts_log = ts_log.tz_localize('UTC')
                                else:
                                    ts_log = ts_log.tz_convert('UTC')
                                ts_log = ts_log.to_pydatetime()
                            add_signal(
                                action=sig.action.value,
                                reason=sig.reason,
                                price=sig.price,
                                timestamp=ts_log,
                                symbol=symbol,
                                strategy_type="flat",
                                signal_id=sig.signal_id if hasattr(sig, 'signal_id') and sig.signal_id else None,
                            )
                        except Exception as e:
                            print(f"[web] ⚠️ Failed to save flat signal to history: {e}")
            
            # ML стратегия - генерируем текущие сигналы (если стратегия включена)
            if settings.enable_ml_strategy:
                try:
                    from bot.ml.strategy_ml import build_ml_signals
                    from pathlib import Path
                    
                    # Определяем, какую модель использовать для данного символа на графике
                    models_dir = Path(__file__).parent.parent.parent / "ml_models"
                    model_type_preference = getattr(settings, "ml_model_type_for_all", None)
                    model_path_for_chart = None
                    
                    # 1) Пытаемся использовать явно выбранную модель, если она подходит символу и (при наличии) типу
                    if settings.ml_model_path:
                        model_path_obj = Path(settings.ml_model_path)
                        if model_path_obj.exists():
                            filename = model_path_obj.name
                            if "_" in filename:
                                parts = filename.replace(".pkl", "").split("_")
                                if len(parts) >= 2 and parts[1] == symbol:
                                    model_type_from_filename = parts[0].lower()
                                    if model_type_preference:
                                        if model_type_from_filename == model_type_preference.lower():
                                            model_path_for_chart = str(model_path_obj)
                                            print(f"[web] Using explicit ML model for chart {symbol}: {model_path_for_chart} (matches type: {model_type_preference})")
                                        else:
                                            print(
                                                f"[web] Explicit ML model for chart {symbol} ({model_type_from_filename}) "
                                                f"doesn't match global preference ({model_type_preference}), ignoring it"
                                            )
                                    else:
                                        model_path_for_chart = str(model_path_obj)
                                        print(f"[web] Using explicit ML model for chart {symbol}: {model_path_for_chart}")
                    
                    # 2) Если явная модель не подходит, ищем по предпочтению или авто (ensemble > rf > xgb)
                    if not model_path_for_chart and models_dir.exists():
                        if model_type_preference:
                            pattern = f"{model_type_preference}_{symbol}_*.pkl"
                            for model_file in sorted(models_dir.glob(pattern), reverse=True):
                                if model_file.is_file():
                                    model_path_for_chart = str(model_file)
                                    print(f"[web] Using preferred ML model for chart {symbol}: {model_path_for_chart}")
                                    break
                        else:
                            for model_type in ["ensemble", "rf", "xgb"]:
                                pattern = f"{model_type}_{symbol}_*.pkl"
                                for model_file in sorted(models_dir.glob(pattern), reverse=True):
                                    if model_file.is_file():
                                        model_path_for_chart = str(model_file)
                                        print(f"[web] Using auto-selected ML model for chart {symbol}: {model_path_for_chart}")
                                        break
                                if model_path_for_chart:
                                    break
                    
                    if model_path_for_chart and len(df_ready) >= 200:
                        ml_signals = build_ml_signals(
                            df_ready,
                            model_path=model_path_for_chart,
                            confidence_threshold=settings.ml_confidence_threshold,
                            min_signal_strength=settings.ml_min_signal_strength,
                            stability_filter=settings.ml_stability_filter,
                            leverage=settings.leverage,
                            target_profit_pct_margin=settings.ml_target_profit_pct_margin,
                            max_loss_pct_margin=settings.ml_max_loss_pct_margin,
                        )
                        # Добавляем только LONG и SHORT сигналы (HOLD не показываем)
                        for sig in ml_signals:
                            if sig.action in (Action.LONG, Action.SHORT):
                                signals.append(sig)
                                # Сохраняем ML сигнал в историю для синхронизации с графиком
                                try:
                                    from bot.web.history import add_signal
                                    ts_log = sig.timestamp
                                    if isinstance(ts_log, pd.Timestamp):
                                        if ts_log.tzinfo is None:
                                            ts_log = ts_log.tz_localize('UTC')
                                        else:
                                            ts_log = ts_log.tz_convert('UTC')
                                        ts_log = ts_log.to_pydatetime()
                                    add_signal(
                                        action=sig.action.value,
                                        reason=sig.reason,
                                        price=sig.price,
                                        timestamp=ts_log,
                                        symbol=symbol,
                                        strategy_type="ml",
                                        signal_id=sig.signal_id if hasattr(sig, 'signal_id') and sig.signal_id else None,
                                    )
                                except Exception as e:
                                    print(f"[web] ⚠️ Failed to save ML signal to history: {e}")
                except Exception as e:
                    print(f"[web] Error generating ML signals for chart: {e}")
            
            # Momentum стратегия - генерируем текущие сигналы (если стратегия включена)
            if settings.enable_momentum_strategy:
                try:
                    momentum_signals = build_signals(df_ready, settings.strategy, use_momentum=True, use_liquidity=False)
                    _web_log(f"[web] Strategy processed")
                    for sig in momentum_signals:
                        # Добавляем только LONG и SHORT сигналы (HOLD не показываем)
                        if sig.reason.startswith("momentum_") and sig.action in (Action.LONG, Action.SHORT):
                            signals.append(sig)
                            # Сохраняем сигнал в историю для синхронизации с графиком
                            try:
                                from bot.web.history import add_signal
                                ts_log = sig.timestamp
                                if isinstance(ts_log, pd.Timestamp):
                                    if ts_log.tzinfo is None:
                                        ts_log = ts_log.tz_localize('UTC')
                                    else:
                                        ts_log = ts_log.tz_convert('UTC')
                                    ts_log = ts_log.to_pydatetime()
                                add_signal(
                                    action=sig.action.value,
                                    reason=sig.reason,
                                    price=sig.price,
                                    timestamp=ts_log,
                                    symbol=symbol,
                                    strategy_type="momentum",
                                    signal_id=sig.signal_id if hasattr(sig, 'signal_id') and sig.signal_id else None,
                                )
                            except Exception as e:
                                print(f"[web] ⚠️ Failed to save momentum signal to history: {e}")
                except Exception as e:
                    print(f"[web] Error generating Momentum signals for chart: {e}")
            
            # Liquidity Sweep стратегия - генерируем текущие сигналы (если стратегия включена)
            if settings.enable_liquidity_sweep_strategy:
                try:
                    liquidity_signals = build_signals(df_ready, settings.strategy, use_momentum=False, use_liquidity=True)
                    _web_log(f"[web] Strategy processed")
                    for sig in liquidity_signals:
                        # Добавляем только LONG и SHORT сигналы (HOLD не показываем)
                        if sig.reason.startswith("liquidity_") and sig.action in (Action.LONG, Action.SHORT):
                            signals.append(sig)
                            # Сохраняем сигнал в историю для синхронизации с графиком
                            try:
                                from bot.web.history import add_signal
                                ts_log = sig.timestamp
                                if isinstance(ts_log, pd.Timestamp):
                                    if ts_log.tzinfo is None:
                                        ts_log = ts_log.tz_localize('UTC')
                                    else:
                                        ts_log = ts_log.tz_convert('UTC')
                                    ts_log = ts_log.to_pydatetime()
                                add_signal(
                                    action=sig.action.value,
                                    reason=sig.reason,
                                    price=sig.price,
                                    timestamp=ts_log,
                                    symbol=symbol,
                                    strategy_type="liquidity",
                                    signal_id=sig.signal_id if hasattr(sig, 'signal_id') and sig.signal_id else None,
                                )
                            except Exception as e:
                                print(f"[web] ⚠️ Failed to save liquidity signal to history: {e}")
                except Exception as e:
                    print(f"[web] Error generating Liquidity signals for chart: {e}")
            
            # SMC стратегия - генерируем текущие сигналы (если стратегия включена)
            if settings.enable_smc_strategy:
                try:
                    from bot.smc_strategy import build_smc_signals
                    smc_signals = build_smc_signals(df_ready, settings.strategy, symbol=symbol)
                    _web_log(f"[web] SMC Strategy processed: {len(smc_signals)} signals")
                    for sig in smc_signals:
                        # Добавляем только LONG и SHORT сигналы (HOLD не показываем)
                        if sig.action in (Action.LONG, Action.SHORT):
                            signals.append(sig)
                            # Сохраняем SMC сигнал в общую историю для синхронизации с графиком
                            try:
                                from bot.web.history import add_signal
                                ts_log = sig.timestamp
                                if isinstance(ts_log, pd.Timestamp):
                                    if ts_log.tzinfo is None:
                                        ts_log = ts_log.tz_localize('UTC')
                                    else:
                                        ts_log = ts_log.tz_convert('UTC')
                                    ts_log = ts_log.to_pydatetime()
                                add_signal(
                                    action=sig.action.value,
                                    reason=sig.reason,
                                    price=sig.price,
                                    timestamp=ts_log,
                                    symbol=symbol,
                                    strategy_type="smc",
                                    signal_id=sig.signal_id if hasattr(sig, 'signal_id') and sig.signal_id else None,
                                )
                            except Exception as e:
                                print(f"[web] ⚠️ Failed to save SMC signal to history: {e}")
                except Exception as e:
                    print(f"[web] Error generating SMC signals for chart: {e}")
            
            if signals is None:
                signals = []
        except Exception as e:
            print(f"[web] Error building signals: {e}")
            import traceback
            print(f"[web] Traceback: {traceback.format_exc()}")
            # Не очищаем signals, оставляем исторические сигналы
        
        # Подготавливаем данные для графика
        candles = []
        times = []
        for idx, row in df_ready.iterrows():
            try:
                # Конвертируем индекс времени в Unix timestamp (миллисекунды)
                if isinstance(idx, pd.Timestamp):
                    # Если время без таймзоны, считаем его UTC (как приходит от Bybit)
                    if idx.tzinfo is None:
                        # Локализуем как UTC
                        utc_time = utc_tz.localize(idx)
                    else:
                        # Конвертируем в UTC
                        utc_time = idx.astimezone(utc_tz)
                    # Unix timestamp в миллисекундах
                    time_ms = int(utc_time.timestamp() * 1000)
                else:
                    # Если не Timestamp, пытаемся преобразовать
                    try:
                        if isinstance(idx, str):
                            # Пытаемся распарсить строку
                            dt = pd.to_datetime(idx)
                            if dt.tzinfo is None:
                                utc_time = utc_tz.localize(dt)
                            else:
                                utc_time = dt.astimezone(utc_tz)
                            time_ms = int(utc_time.timestamp() * 1000)
                        else:
                            time_ms = int(idx) if isinstance(idx, (int, float)) else 0
                    except:
                        time_ms = 0
                
                candles.append({
                    "time": time_ms,  # Unix timestamp в миллисекундах
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                })
                times.append(time_ms)
            except Exception as e:
                print(f"[web] Error processing candle: {e}")
                continue
        
        if not candles:
            return jsonify({"error": "No candles data available"}), 500
        
        # Индикаторы (с проверкой наличия колонок)
        num_candles = len(candles)
        def safe_get_column(df, col_name, default=None):
            """Безопасное получение колонки с дефолтным значением."""
            if col_name in df.columns:
                try:
                    values = [float(x) if pd.notna(x) else None for x in df[col_name].values]
                    # Выравниваем длину с количеством свечей
                    if len(values) != num_candles:
                        if len(values) > num_candles:
                            values = values[-num_candles:]
                        else:
                            values = [default] * (num_candles - len(values)) + values
                    return values
                except Exception as e:
                    print(f"[web] Error getting column {col_name}: {e}")
                    return [default] * num_candles if default is not None else None
            return [default] * num_candles if default is not None else None
        
        # Безопасное определение фазы рынка
        market_phases = []
        if "adx" in df_ready.columns:
            try:
                for _, row in df_ready.iterrows():
                    try:
                        phase = detect_market_phase(row, settings.strategy)
                        market_phases.append(phase.value if phase else "unknown")
                    except Exception:
                        market_phases.append("unknown")
            except Exception:
                market_phases = ["unknown"] * num_candles
        else:
            market_phases = ["unknown"] * num_candles
        
        # Выравниваем длину market_phases
        if len(market_phases) != num_candles:
            if len(market_phases) > num_candles:
                market_phases = market_phases[-num_candles:]
            else:
                market_phases = ["unknown"] * (num_candles - len(market_phases)) + market_phases
        
        indicators = {
            "sma": safe_get_column(df_ready, "sma"),
            "bb_upper": safe_get_column(df_ready, "bb_upper"),
            "bb_middle": safe_get_column(df_ready, "bb_middle"),
            "bb_lower": safe_get_column(df_ready, "bb_lower"),
            "rsi": safe_get_column(df_ready, "rsi"),
            "adx": safe_get_column(df_ready, "adx"),
            "macd": safe_get_column(df_ready, "macd"),
            "macd_signal": safe_get_column(df_ready, "macd_signal"),
            "macd_hist": safe_get_column(df_ready, "macd_hist"),
            "market_phase": market_phases,
        }
        
        # Сигналы (точки входа) - конвертируем время в Unix timestamp (миллисекунды) для графика
        entry_signals = []
        print(f"[web] Processing signals for chart: total={len(signals) if signals else 0}")
        if signals and isinstance(signals, list) and len(signals) > 0:
            for sig in signals:
                try:
                    if sig and hasattr(sig, 'action') and sig.action and hasattr(sig.action, 'value'):
                        action_value = sig.action.value.lower()
                        # Проверяем, является ли сигнал actionable для графика
                        if action_value in ("long", "short"):
                            timestamp_ms = None
                            if hasattr(sig, 'timestamp') and sig.timestamp:
                                try:
                                    # Конвертируем в Unix timestamp (миллисекунды)
                                    if isinstance(sig.timestamp, pd.Timestamp):
                                        if sig.timestamp.tzinfo is None:
                                            utc_time = sig.timestamp.tz_localize('UTC')
                                        else:
                                            utc_time = sig.timestamp.tz_convert('UTC')
                                        timestamp_ms = int(utc_time.timestamp() * 1000)
                                    elif hasattr(sig.timestamp, 'timestamp'):
                                        # datetime объект
                                        timestamp_ms = int(sig.timestamp.timestamp() * 1000)
                                    else:
                                        # Строка - парсим
                                        dt = pd.to_datetime(str(sig.timestamp))
                                        if dt.tzinfo is None:
                                            dt = dt.tz_localize('UTC')
                                        timestamp_ms = int(dt.timestamp() * 1000)
                                except Exception as e:
                                    print(f"[web] Error converting signal timestamp: {e}")
                                    timestamp_ms = None
                            
                            if timestamp_ms is None:
                                continue  # Пропускаем сигнал без валидного времени
                            
                            signal_price = float(getattr(sig, 'price', 0)) if hasattr(sig, 'price') and sig.price else 0.0
                            signal_reason = getattr(sig, 'reason', '') or ""
                            
                            entry_signals.append({
                                "time": timestamp_ms,  # Unix timestamp в миллисекундах
                                "action": action_value,
                                "reason": signal_reason,
                                "price": signal_price,
                            })
                except Exception as e:
                    print(f"[web] Error processing signal: {e}")
                    continue
            
            if entry_signals:
                print(f"[web] ✅ Chart signals: {len(entry_signals)} (LONG: {sum(1 for s in entry_signals if s['action']=='long')}, SHORT: {sum(1 for s in entry_signals if s['action']=='short')})")
                # Показываем первые 3 сигнала для отладки
                for i, sig in enumerate(entry_signals[:3]):
                    print(f"[web]   Signal {i+1}: {sig['action'].upper()} @ ${sig['price']:.2f}, time_ms={sig['time']}, reason={sig['reason'][:30]}")
            else:
                print(f"[web] ⚠️ No valid entry signals for chart (from {len(signals) if signals else 0} total signals)")
        
        # Текущая фаза рынка (последний бар)
        current_phase = indicators["market_phase"][-1] if indicators["market_phase"] and len(indicators["market_phase"]) > 0 else "unknown"
        current_adx = indicators["adx"][-1] if indicators["adx"] and len(indicators["adx"]) > 0 and indicators["adx"][-1] is not None else None
        
        # Информация о текущей позиции
        position_info = None
        if position:
            try:
                tp_val = position.get("takeProfit") or position.get("take_profit")
                sl_val = position.get("stopLoss") or position.get("stop_loss")
                
                position_info = {
                    "side": position["side"],
                    "size": float(position["size"]),
                    "entry_price": float(position["avg_price"]),
                    "unrealised_pnl": float(position["unrealised_pnl"]),
                    "take_profit": float(tp_val) if tp_val and str(tp_val).strip() != "0" else 0.0,
                    "stop_loss": float(sl_val) if sl_val and str(sl_val).strip() != "0" else 0.0,
                }
            except Exception as e:
                print(f"[web] Error formatting position info: {e}")
                # Fallback к сырым данным если конвертация не удалась
                position_info = {
                    "side": position.get("side", ""),
                    "size": position.get("size", 0),
                    "entry_price": position.get("avg_price", 0),
                    "unrealised_pnl": position.get("unrealised_pnl", 0),
                    "take_profit": 0.0,
                    "stop_loss": 0.0,
                }
        
        return jsonify({
            "candles": candles,
            "indicators": indicators,
            "signals": entry_signals,
            "times": times,
            "market_phase": current_phase,
            "current_adx": current_adx,
            "position": position_info,
            "symbol": symbol,  # Добавляем информацию о символе
        })
    except Exception as e:
        import traceback
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"[web] Error in api_chart_data: {error_msg}")
        print(f"[web] Traceback: {error_trace}")
        # Возвращаем только сообщение об ошибке, без traceback для безопасности
        return jsonify({"error": f"Error loading chart data: {error_msg}"}), 500


# Matplotlib endpoint удален - используем только Plotly.js

@app.route("/api/bybit/settings")
@login_required
def api_get_bybit_settings():
    """Получить текущие настройки API Bybit."""
    if not settings:
        return jsonify({"error": "Settings not loaded"}), 500
    
    api_key = settings.api.api_key or ""
    api_secret = settings.api.api_secret or ""
    base_url = settings.api.base_url or "https://api.bybit.com"
    
    # Возвращаем полные значения для редактирования (пользователь уже авторизован)
    return jsonify({
        "api_key": api_key,
        "api_secret": api_secret,
        "base_url": base_url
    })


@app.route("/api/admin/change-password", methods=["POST"])
@login_required
def api_change_admin_password():
    """Изменить пароль администратора."""
    global ADMIN_PASSWORD
    
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    try:
        current_password = data.get("current_password", "").strip()
        new_password = data.get("new_password", "").strip()
        
        if not current_password or not new_password:
            return jsonify({"error": "Both current_password and new_password are required"}), 400
        
        if len(new_password) < 6:
            return jsonify({"error": "New password must be at least 6 characters long"}), 400
        
        # Проверяем текущий пароль
        if current_password != ADMIN_PASSWORD:
            return jsonify({"error": "Current password is incorrect"}), 401
        
        # Сохраняем новый пароль в .env файл
        env_path = Path(__file__).parent.parent.parent / ".env"
        
        # Читаем существующий .env файл
        env_lines = []
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8') as f:
                env_lines = f.readlines()
        
        # Создаем словарь существующих переменных
        env_dict = {}
        keys_to_update = {'ADMIN_USERNAME', 'ADMIN_PASSWORD'}
        
        # Собираем все существующие переменные, пропуская те, которые будем обновлять
        for line in env_lines:
            line_stripped = line.strip()
            if line_stripped and not line_stripped.startswith('#') and '=' in line_stripped:
                key, value = line_stripped.split('=', 1)
                key = key.strip()
                if key not in keys_to_update:
                    env_dict[key] = value.strip()
        
        # Обновляем настройки администратора
        env_dict['ADMIN_USERNAME'] = ADMIN_USERNAME
        env_dict['ADMIN_PASSWORD'] = new_password
        
        # Сохраняем обратно в .env
        with open(env_path, 'w', encoding='utf-8') as f:
            # Сохраняем все существующие строки (кроме обновляемых)
            admin_section_found = False
            for line in env_lines:
                line_stripped = line.strip()
                # Пропускаем строки с настройками администратора
                if line_stripped and not line_stripped.startswith('#') and '=' in line_stripped:
                    key = line_stripped.split('=', 1)[0].strip()
                    if key in keys_to_update:
                        continue  # Пропускаем старые значения
                
                # Пропускаем комментарий о настройках администратора если он есть
                if "Admin" in line or "ADMIN" in line or "auto-updated by admin panel" in line:
                    admin_section_found = True
                    continue
                
                f.write(line)
            
            # Добавляем обновленные настройки в конец
            if not admin_section_found:
                f.write(f"\n# Admin authentication settings (auto-updated by admin panel)\n")
            f.write(f"ADMIN_USERNAME={env_dict['ADMIN_USERNAME']}\n")
            f.write(f"ADMIN_PASSWORD={env_dict['ADMIN_PASSWORD']}\n")
        
        # Обновляем глобальную переменную (для текущей сессии)
        ADMIN_PASSWORD = new_password
        
        # Также обновляем переменную окружения
        import os
        os.environ['ADMIN_PASSWORD'] = new_password
        
        print(f"[web] Admin password updated successfully")
        
        return jsonify({
            "success": True,
            "message": "Password changed successfully. Please log in again."
        })
        
    except Exception as e:
        print(f"[web] Error changing admin password: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/bybit/settings", methods=["POST"])
@login_required
def api_update_bybit_settings():
    """Обновить настройки API Bybit и сохранить в .env файл."""
    global settings, client
    
    if not settings:
        return jsonify({"error": "Settings not loaded"}), 500
    
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    try:
        api_key = data.get("api_key", "").strip()
        api_secret = data.get("api_secret", "").strip()
        base_url = data.get("base_url", "").strip()
        
        if not api_key or not api_secret or not base_url:
            return jsonify({"error": "All fields (api_key, api_secret, base_url) are required"}), 400
        
        # Проверяем валидность base_url
        valid_urls = ["https://api.bybit.com", "https://api-testnet.bybit.com"]
        if base_url not in valid_urls:
            return jsonify({"error": f"Invalid base_url. Must be one of: {', '.join(valid_urls)}"}), 400
        
        # Обновляем настройки
        settings.api.api_key = api_key
        settings.api.api_secret = api_secret
        settings.api.base_url = base_url
        
        # Сохраняем в .env файл
        env_path = Path(__file__).parent.parent.parent / ".env"
        
        # Читаем существующий .env файл
        env_lines = []
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8') as f:
                env_lines = f.readlines()
        
        # Создаем словарь существующих переменных
        env_dict = {}
        keys_to_update = {'BYBIT_API_KEY', 'BYBIT_API_SECRET', 'BYBIT_BASE_URL'}
        
        # Собираем все существующие переменные, пропуская те, которые будем обновлять
        for line in env_lines:
            line_stripped = line.strip()
            if line_stripped and not line_stripped.startswith('#') and '=' in line_stripped:
                key, value = line_stripped.split('=', 1)
                key = key.strip()
                if key not in keys_to_update:
                    env_dict[key] = value.strip()
        
        # Обновляем API настройки
        env_dict['BYBIT_API_KEY'] = api_key
        env_dict['BYBIT_API_SECRET'] = api_secret
        env_dict['BYBIT_BASE_URL'] = base_url
        
        # Сохраняем обратно в .env
        with open(env_path, 'w', encoding='utf-8') as f:
            # Сохраняем все существующие строки (кроме обновляемых)
            api_section_found = False
            for line in env_lines:
                line_stripped = line.strip()
                # Пропускаем строки с API настройками
                if line_stripped and not line_stripped.startswith('#') and '=' in line_stripped:
                    key = line_stripped.split('=', 1)[0].strip()
                    if key in keys_to_update:
                        continue  # Пропускаем старые значения
                
                # Пропускаем комментарий о API настройках если он есть
                if "Bybit API" in line or "API settings" in line or "auto-updated by admin panel" in line:
                    api_section_found = True
                    continue
                
                f.write(line)
            
            # Добавляем обновленные настройки в конец
            if not api_section_found:
                f.write(f"\n# Bybit API settings (auto-updated by admin panel)\n")
            f.write(f"BYBIT_API_KEY={env_dict['BYBIT_API_KEY']}\n")
            f.write(f"BYBIT_API_SECRET={env_dict['BYBIT_API_SECRET']}\n")
            f.write(f"BYBIT_BASE_URL={env_dict['BYBIT_BASE_URL']}\n")
        
        # Обновляем клиент с новыми настройками
        client = BybitClient(settings.api)
        
        # Обновляем настройки в shared_settings для работающего бота
        from bot.shared_settings import set_settings
        set_settings(settings)
        
        print(f"[web] Bybit API settings updated. Base URL: {base_url}")
        
        return jsonify({
            "success": True,
            "message": "API settings saved successfully"
        })
        
    except Exception as e:
        print(f"[web] Error updating Bybit API settings: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def run_web_server(host="127.0.0.1", port=5000, debug=False):
    """Запустить веб-сервер."""
    init_app()
    
    # Отключаем мусорные логи waitress (Task queue depth is X)
    import logging
    logging.getLogger('waitress.queue').setLevel(logging.ERROR)
    
    if debug:
        # В режиме отладки используем встроенный сервер Flask
        app.run(host=host, port=port, debug=debug)
    else:
        # В продакшене используем gunicorn (должен быть запущен отдельно)
        # Или waitress для кроссплатформенности
        try:
            from waitress import serve
            print(f"[web] Starting production server with Waitress on {host}:{port}")
            serve(app, host=host, port=port, threads=4)
        except ImportError:
            print("[web] Waitress not installed, falling back to Flask development server")
            print("[web] WARNING: Flask development server is NOT suitable for production!")
            print("[web] Install waitress: pip install waitress")
            print("[web] Or use gunicorn: gunicorn -w 4 -b {host}:{port} 'bot.web.app:app'")
            app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    run_web_server(debug=True)

