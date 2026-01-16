"""
Менеджер для управления несколькими торговыми ботами одновременно.
Каждая торговая пара работает в отдельном потоке.
"""
import threading
import time
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field

from bot.config import AppSettings
from bot.live import run_live_from_api
from bot.shared_settings import get_settings, set_settings
from bot.exchange.bybit_client import BybitClient

# Глобальный словарь для обновления статуса воркеров из run_live_from_api
# Формат: {symbol: {"last_update": timestamp, "error": error_msg}}
_worker_status_updates: Dict[str, Dict[str, Any]] = {}
_worker_status_lock = threading.Lock()


def update_worker_status(symbol: str, **kwargs):
    """
    Обновить статус воркера из run_live_from_api.
    Используется для мониторинга состояния воркеров.
    
    Args:
        symbol: Торговая пара
        **kwargs: Параметры для обновления (last_update, error, etc.)
    """
    with _worker_status_lock:
        if symbol not in _worker_status_updates:
            _worker_status_updates[symbol] = {}
        _worker_status_updates[symbol].update(kwargs)
        _worker_status_updates[symbol]["last_update"] = time.time()


def get_worker_status_updates() -> Dict[str, Dict[str, Any]]:
    """Получить все обновления статуса воркеров."""
    with _worker_status_lock:
        return _worker_status_updates.copy()


@dataclass
class WorkerStatus:
    """Статус воркера для одной торговой пары"""
    symbol: str
    running: bool = False
    thread: Optional[threading.Thread] = None
    error: Optional[str] = None
    last_update: Optional[float] = None
    settings: Optional[AppSettings] = None
    restart_count: int = 0  # Количество рестартов этого воркера
    last_restart_time: Optional[float] = None  # Время последнего рестарта
    consecutive_failures: int = 0  # Количество последовательных падений
    stop_event: threading.Event = field(default_factory=threading.Event)  # Событие для остановки воркера


class MultiSymbolManager:
    """Управляет несколькими торговыми ботами одновременно"""
    
    def __init__(self, settings: AppSettings):
        """
        Инициализация менеджера.
        
        Args:
            settings: Настройки приложения с активными символами
        """
        self.settings = settings
        self.workers: Dict[str, WorkerStatus] = {}  # {symbol: WorkerStatus}
        self.running = False
        self.lock = threading.Lock()
        self.monitor_thread: Optional[threading.Thread] = None
        self.monitor_interval = 30.0  # Интервал проверки состояния воркеров (секунды)
        self.max_restarts = 5  # Максимальное количество рестартов для одного воркера
        self.restart_delay = 10.0  # Задержка перед рестартом (секунды)
        self.worker_timeout = 300.0  # Увеличиваем таймаут до 5 минут (для тяжелых вычислений ML/SMC)
        # Кэш для моделей (чтобы не искать каждый раз)
        self._model_cache: Dict[str, Optional[str]] = {}
        self._model_cache_keys: Dict[str, str] = {}  # Ключи кэша для отслеживания изменений
        # Инициализируем воркеры при создании (lock не нужен, так как объект еще не используется)
        with self.lock:
            self._initialize_workers()
    
    def _initialize_workers(self):
        """
        Инициализирует воркеры для всех активных символов.
        ВАЖНО: Эта функция НЕ захватывает self.lock, предполагается, что вызывающая функция уже держит блокировку.
        """
        # Создаем воркеры для всех активных символов
        for symbol in self.settings.active_symbols:
            if symbol not in self.workers:
                print(f"[MultiSymbol] ⚙️  Creating worker for new symbol: {symbol}")
                self.workers[symbol] = WorkerStatus(
                    symbol=symbol,
                    running=False,
                    settings=self._create_settings_for_symbol(symbol),
                    stop_event=threading.Event()
                )
                print(f"[MultiSymbol] ⚙️  Worker created for {symbol}")
            else:
                print(f"[MultiSymbol] ⚙️  Worker already exists for {symbol}")
                # СБРОС СОБЫТИЯ ОСТАНОВКИ (ВАЖНО для перезапуска!)
                if hasattr(self.workers[symbol], 'stop_event'):
                    self.workers[symbol].stop_event.clear()
                    print(f"[MultiSymbol] ⚙️  Stop event cleared for {symbol}")
    
    def _detect_and_add_open_positions(self) -> Set[str]:
        """
        Обнаруживает все открытые позиции на Bybit и автоматически добавляет их в управление.
        Возвращает множество символов с открытыми позициями.
        
        ВАЖНО: Эта функция НЕ захватывает self.lock.
        """
        detected_symbols: Set[str] = set()
        
        try:
            # Создаем клиент Bybit
            client = BybitClient(self.settings.api)
            
            # Получаем ВСЕ открытые позиции по USDT
            print("[MultiSymbol] 🔍 Scanning Bybit for ALL open positions...")
            response = client.get_position_info(settle_coin="USDT")
            
            if response.get("retCode") != 0:
                print(f"[MultiSymbol] ⚠️ Failed to get positions: {response.get('retMsg', 'Unknown error')}")
                return detected_symbols
            
            positions = response.get("result", {}).get("list", [])
            
            for pos in positions:
                size = float(pos.get("size", 0))
                if size > 0:  # Есть открытая позиция
                    symbol = pos.get("symbol", "")
                    side = pos.get("side", "")
                    avg_price = float(pos.get("avgPrice", 0))
                    unrealised_pnl = float(pos.get("unrealisedPnl", 0))
                    
                    detected_symbols.add(symbol)
                    
                    print(f"[MultiSymbol] 📊 Found open position: {symbol} {side} size={size} entry=${avg_price:.2f} PnL=${unrealised_pnl:.2f}")
                    
                    # Добавляем символ в active_symbols, если его там нет
                    if symbol not in self.settings.active_symbols:
                        print(f"[MultiSymbol] ➕ Auto-adding {symbol} to active symbols (has open position)")
                        self.settings.active_symbols.append(symbol)
                    
                    # Создаем воркер, если его нет
                    if symbol not in self.workers:
                        print(f"[MultiSymbol] ⚙️  Creating worker for {symbol} (auto-detected position)")
                        self.workers[symbol] = WorkerStatus(
                            symbol=symbol,
                            running=False,
                            settings=self._create_settings_for_symbol(symbol),
                            stop_event=threading.Event()
                        )
            
            if detected_symbols:
                print(f"[MultiSymbol] ✅ Auto-detected {len(detected_symbols)} symbols with open positions: {detected_symbols}")
            else:
                print("[MultiSymbol] ℹ️  No open positions found on Bybit")
                
        except Exception as e:
            print(f"[MultiSymbol] ⚠️ Error detecting open positions: {e}")
            import traceback
            traceback.print_exc()
        
        return detected_symbols
    
    def _create_settings_for_symbol(self, symbol: str) -> AppSettings:
        """
        Создает копию настроек для конкретного символа.
        Оптимизировано: использует shallow copy для базовых настроек вместо deepcopy.
        
        Args:
            symbol: Торговая пара
        
        Returns:
            Копия AppSettings с переопределенным символом
        """
        # Используем dataclasses.replace для создания копии (быстрее чем deepcopy)
        from dataclasses import replace
        
        try:
            # Создаем новый объект с теми же значениями, но с переопределенным символом
            # ВАЖНО: primary_symbol НЕ переопределяем - он должен оставаться глобальным PRIMARY_SYMBOL
            symbol_settings = replace(
                self.settings,
                symbol=symbol,
                # primary_symbol остается из self.settings (глобальный PRIMARY_SYMBOL)
            )
        except Exception as e:
            # Fallback на deepcopy, если replace не работает
            print(f"[MultiSymbol] ⚠️  Warning: replace() failed for {symbol}, using deepcopy: {e}")
            import copy
            symbol_settings = copy.deepcopy(self.settings)
            symbol_settings.symbol = symbol
            # primary_symbol остается из self.settings (глобальный PRIMARY_SYMBOL)
        
        # Автоматически находим ML модель для символа, если ML стратегия включена
                # ВАЖНО: Всегда переопределяем ml_model_path для каждого символа, даже если он уже установлен
        if symbol_settings.enable_ml_strategy:
            import pathlib
            models_dir = pathlib.Path(__file__).parent.parent / "ml_models"
            print(f"[MultiSymbol] 🔍 Searching for ML model for {symbol} in {models_dir}")
            
            if models_dir.exists():
                
                # ВАЖНО: Всегда проверяем предпочтение типа модели из глобальных настроек
                # и очищаем кэш, если настройка изменилась
                model_type_preference = getattr(self.settings, 'ml_model_type_for_all', None)
                
                # Формируем ключ кэша с учетом предпочтения типа модели и явно выбранной модели
                explicit_model_path = getattr(self.settings, 'ml_model_path', None)
                cache_key = f"{symbol}_{model_type_preference or 'auto'}_{explicit_model_path or 'none'}"
                if not hasattr(self, '_model_cache_keys'):
                    self._model_cache_keys = {}
                
                # Если ключ кэша изменился, очищаем кэш для этого символа
                if symbol in self._model_cache_keys and self._model_cache_keys.get(symbol) != cache_key:
                    print(f"[MultiSymbol] 🔄 Model selection changed for {symbol}, clearing cache")
                    print(f"[MultiSymbol]    Old key: {self._model_cache_keys.get(symbol)}")
                    print(f"[MultiSymbol]    New key: {cache_key}")
                    if symbol in self._model_cache:
                        del self._model_cache[symbol]
                
                if symbol not in self._model_cache:
                    # Ищем модель для символа с учетом предпочтения типа модели
                    found_model = None
                    
                    # СНАЧАЛА: Проверяем, есть ли явно выбранная модель в settings.ml_model_path
                    # и соответствует ли она текущему символу И типу модели (если ml_model_type_for_all задан)
                    if self.settings.ml_model_path:
                        explicit_model_path = pathlib.Path(self.settings.ml_model_path)
                        if explicit_model_path.exists():
                            # Извлекаем символ и тип модели из имени файла
                            model_filename = explicit_model_path.name
                            # Формат: ensemble_BTCUSDT_15.pkl или rf_ETHUSDT_15.pkl
                            if "_" in model_filename:
                                parts = model_filename.replace('.pkl', '').split('_')
                                if len(parts) >= 2 and parts[1] == symbol:
                                    # Модель соответствует текущему символу
                                    # Теперь проверяем, соответствует ли она типу модели из ml_model_type_for_all
                                    model_type_from_filename = parts[0].lower()  # ensemble, rf, xgb
                                    
                                    # Если ml_model_type_for_all задан, проверяем соответствие
                                    if model_type_preference:
                                        if model_type_from_filename == model_type_preference.lower():
                                            # Модель соответствует и символу, и типу - используем её
                                            found_model = str(explicit_model_path)
                                            print(f"[MultiSymbol] ✅ Using explicitly selected model for {symbol}: {found_model} (matches type: {model_type_preference})")
                                        else:
                                            # Модель соответствует символу, но не типу - игнорируем её
                                            # Убрано verbose сообщение о несовпадении модели - это нормальное поведение
                                            pass
                                    else:
                                        # ml_model_type_for_all не задан - используем явно выбранную модель
                                        found_model = str(explicit_model_path)
                                        print(f"[MultiSymbol] ✅ Using explicitly selected model for {symbol}: {found_model}")
                    
                    # ЕСЛИ явно выбранная модель не найдена или не соответствует символу/типу, ищем автоматически
                    if not found_model:
                        if model_type_preference:
                            # Если задан тип модели, ищем только этот тип
                            pattern = f"{model_type_preference}_{symbol}_*.pkl"
                            print(f"[MultiSymbol] 🔍 Looking for {model_type_preference.upper()} models matching: {pattern} (user preference: {model_type_preference})")
                            for model_file in sorted(models_dir.glob(pattern), reverse=True):  # Новые модели первыми
                                if model_file.is_file():
                                    found_model = str(model_file)
                                    print(f"[MultiSymbol] ✅ Found {model_type_preference.upper()} model: {found_model}")
                                    break
                        else:
                            # Автоматический выбор: предпочитаем ensemble > rf > xgb
                            # Сначала ищем ensemble
                            ensemble_pattern = f"ensemble_{symbol}_*.pkl"
                            print(f"[MultiSymbol] 🔍 Auto-selection: Looking for Ensemble models matching: {ensemble_pattern}")
                            for model_file in sorted(models_dir.glob(ensemble_pattern), reverse=True):  # Новые модели первыми
                                if model_file.is_file():
                                    found_model = str(model_file)
                                    print(f"[MultiSymbol] ✅ Found Ensemble model: {found_model}")
                                    break
                            
                            # Если ensemble не найден, пробуем rf_
                            if not found_model:
                                rf_pattern = f"rf_{symbol}_*.pkl"
                                print(f"[MultiSymbol] 🔍 Ensemble not found, looking for RF models matching: {rf_pattern}")
                                for model_file in sorted(models_dir.glob(rf_pattern), reverse=True):  # Новые модели первыми
                                    if model_file.is_file():
                                        found_model = str(model_file)
                                        print(f"[MultiSymbol] ✅ Found RF model: {found_model}")
                                        break
                            
                            # Если rf_ модель не найдена, пробуем xgb_
                            if not found_model:
                                xgb_pattern = f"xgb_{symbol}_*.pkl"
                                print(f"[MultiSymbol] 🔍 RF model not found, looking for XGB models matching: {xgb_pattern}")
                                for model_file in sorted(models_dir.glob(xgb_pattern), reverse=True):  # Новые модели первыми
                                    if model_file.is_file():
                                        found_model = str(model_file)
                                        print(f"[MultiSymbol] ✅ Found XGB model: {found_model}")
                                        break
                    
                    if not found_model:
                        print(f"[MultiSymbol] ❌ No ML model found for {symbol}")
                        if model_type_preference:
                            print(f"[MultiSymbol]    Searched for: {model_type_preference}_{symbol}_*.pkl")
                        else:
                            print(f"[MultiSymbol]    Searched for: ensemble_{symbol}_*.pkl, rf_{symbol}_*.pkl, xgb_{symbol}_*.pkl")
                    
                    self._model_cache[symbol] = found_model
                    self._model_cache_keys[symbol] = cache_key
                
                if self._model_cache.get(symbol):
                    symbol_settings.ml_model_path = self._model_cache[symbol]
                    print(f"[MultiSymbol] ✅ Auto-selected ML model for {symbol}: {symbol_settings.ml_model_path}")
                else:
                    # Если модель не найдена, отключаем ML стратегию для этого символа
                    print(f"[MultiSymbol] ⚠️  Warning: No ML model found for {symbol}, disabling ML strategy")
                    symbol_settings.enable_ml_strategy = False
                    symbol_settings.ml_model_path = None
            else:
                print(f"[MultiSymbol] ⚠️  Models directory does not exist: {models_dir}")
                symbol_settings.enable_ml_strategy = False
                symbol_settings.ml_model_path = None
        else:
            # Если ML стратегия отключена, но ml_model_path установлен, очищаем его
            if symbol_settings.ml_model_path:
                print(f"[MultiSymbol] ℹ️  ML strategy disabled for {symbol}, clearing ml_model_path")
                symbol_settings.ml_model_path = None
        
        return symbol_settings
    
    def _worker_thread(self, symbol: str, settings: AppSettings):
        """
        Функция воркера для одной торговой пары.
        
        Args:
            symbol: Торговая пара
            settings: Настройки для этого символа
        """
        worker = self.workers.get(symbol)
        if not worker:
            print(f"[MultiSymbol] ⚠️ Worker for {symbol} not found")
            return
        
        try:
            print(f"[MultiSymbol] 🚀 Starting worker for {symbol}")
            print(f"[MultiSymbol]   Settings: symbol={settings.symbol}, active_symbols={settings.active_symbols if hasattr(settings, 'active_symbols') else 'N/A'}")
            print(f"[MultiSymbol]   Strategies: Trend={settings.enable_trend_strategy}, Flat={settings.enable_flat_strategy}, ML={settings.enable_ml_strategy}, Momentum={settings.enable_momentum_strategy}, Liquidity={settings.enable_liquidity_sweep_strategy}, SMC={settings.enable_smc_strategy}, ICT={settings.enable_ict_strategy}, LiquidationHunter={settings.enable_liquidation_hunter_strategy}, ZScore={settings.enable_zscore_strategy}, VBO={settings.enable_vbo_strategy}")
            
            worker.running = True
            worker.last_update = time.time()
            worker.error = None
            
            # Обновляем статус воркера для мониторинга
            update_worker_status(symbol, error=None)
            
            # Запускаем торговый бот для этого символа
            # Передаем settings с переопределенным символом и явно указываем symbol
            print(f"[MultiSymbol] 📞 Calling run_live_from_api for {symbol}...")
            run_live_from_api(
                initial_settings=settings,
                bot_state=None,
                signal_max_age_seconds=60,
                symbol=symbol,  # Явно передаем symbol для этого воркера
                stop_event=worker.stop_event  # Передаем событие остановки
            )
            # Функция вернулась - это нормально при остановке через stop_event
            print(f"[MultiSymbol] ✅ Worker loop ended for {symbol} (normal shutdown)")
        except KeyboardInterrupt:
            # Нормальное завершение по Ctrl+C
            print(f"[MultiSymbol] 🛑 Worker for {symbol} interrupted")
            worker.running = False
            worker.last_update = time.time()
            update_worker_status(symbol, error="Interrupted")
            raise  # Пробрасываем дальше для корректной остановки
        except Exception as e:
            error_msg = str(e)
            print(f"[MultiSymbol] ❌ Error in worker for {symbol}: {error_msg}")
            import traceback
            error_trace = traceback.format_exc()
            print(f"[MultiSymbol] Full traceback for {symbol}:\n{error_trace}")
            worker.error = error_msg
            worker.running = False
            worker.last_update = time.time()
            update_worker_status(symbol, error=error_msg)
        finally:
            worker.running = False
            worker.last_update = time.time()
            print(f"[MultiSymbol] 🛑 Worker for {symbol} stopped (finally block)")
            update_worker_status(symbol, error="Stopped")
    
    def start(self):
        """Запустить торговлю для всех активных символов"""
        with self.lock:
            if self.running:
                print("[MultiSymbol] ⚠️ Manager is already running")
                return
            
            # 🔍 СНАЧАЛА: Автоматически обнаруживаем и добавляем символы с открытыми позициями
            print("[MultiSymbol] 🔍 Step 1: Auto-detecting open positions on Bybit...")
            try:
                detected_symbols = self._detect_and_add_open_positions()
                if detected_symbols:
                    print(f"[MultiSymbol] ✅ Will manage {len(detected_symbols)} symbols with open positions: {detected_symbols}")
            except Exception as e:
                print(f"[MultiSymbol] ⚠️ Position detection failed (continuing anyway): {e}")
            
            if not self.settings.active_symbols or len(self.settings.active_symbols) == 0:
                print("[MultiSymbol] ❌ Error: No active symbols configured and no open positions found")
                raise ValueError("No active symbols configured. Please configure at least one symbol.")
            
            self.running = True
            print(f"[MultiSymbol] 🚀 Starting MultiSymbolManager for symbols: {self.settings.active_symbols}")
            
            # Инициализируем воркеры для активных символов
            try:
                self._initialize_workers()
                print(f"[MultiSymbol] ✅ Initialized {len(self.workers)} workers")
            except Exception as e:
                print(f"[MultiSymbol] ❌ Error initializing workers: {e}")
                import traceback
                traceback.print_exc()
                self.running = False
                raise
            
            # Запускаем воркеры с задержкой для распределения нагрузки на API
            # Bybit rate limits: Order API - 50 req/s, Other API - 20 req/s
            # Добавляем задержку 0.5 секунды между запуском воркеров
            worker_start_delay = 0.5  # секунды между запусками воркеров
            started_count = 0
            
            for idx, symbol in enumerate(self.settings.active_symbols):
                try:
                    worker = self.workers.get(symbol)
                    if not worker:
                        print(f"[MultiSymbol] ⚠️ Worker for {symbol} not found, creating...")
                        self.workers[symbol] = WorkerStatus(
                            symbol=symbol,
                            running=False,
                            settings=self._create_settings_for_symbol(symbol)
                        )
                        worker = self.workers[symbol]
                    
                    if not worker.running:
                        # Добавляем задержку перед запуском (кроме первого воркера)
                        if idx > 0:
                            time.sleep(worker_start_delay)
                            print(f"[MultiSymbol] ⏱️  Rate limit: waiting {worker_start_delay}s before starting {symbol} worker...")
                        
                        # Создаем поток для воркера
                        try:
                            worker.settings = self._create_settings_for_symbol(symbol)
                            worker.thread = threading.Thread(
                                target=self._worker_thread,
                                args=(symbol, worker.settings),
                                name=f"BotWorker-{symbol}",
                                daemon=True
                            )
                            # Сохраняем время запуска потока для мониторинга
                            worker.thread._start_time = time.time()
                            worker.thread.start()
                            
                            # Даем немного времени потоку на запуск, чтобы убедиться, что он запустился
                            time.sleep(0.1)
                            
                            if worker.thread.is_alive():
                                started_count += 1
                                print(f"[MultiSymbol] ✅ Started worker thread for {symbol} (thread ID: {worker.thread.ident}, alive: {worker.thread.is_alive()})")
                            else:
                                print(f"[MultiSymbol] ❌ Worker thread for {symbol} died immediately after start!")
                                worker.error = "Thread died immediately after start"
                                worker.running = False
                        except Exception as e:
                            print(f"[MultiSymbol] ❌ Error creating/starting thread for {symbol}: {e}")
                            import traceback
                            traceback.print_exc()
                            worker.error = str(e)
                            worker.running = False
                    elif worker and worker.running:
                        print(f"[MultiSymbol] ⚠️ Worker for {symbol} is already running")
                except Exception as e:
                    print(f"[MultiSymbol] ❌ Error processing worker for {symbol}: {e}")
                    import traceback
                    traceback.print_exc()
            
            if started_count == 0:
                print("[MultiSymbol] ❌ ERROR: No workers were started!")
                self.running = False
                raise RuntimeError("No workers were started. Check logs for errors.")
            
            print(f"[MultiSymbol] ✅ Started {started_count} worker(s) out of {len(self.settings.active_symbols)} active symbol(s)")
            
            # Даем немного времени воркерам на инициализацию
            print("[MultiSymbol] ⏳ Waiting for workers to initialize (2 seconds)...")
            time.sleep(2.0)
            
            # Проверяем, что воркеры действительно запущены и работают
            active_workers = []
            for symbol, worker in self.workers.items():
                if symbol in self.settings.active_symbols:
                    is_alive = worker.thread and worker.thread.is_alive() if worker.thread else False
                    if is_alive:
                        active_workers.append(symbol)
                        print(f"[MultiSymbol] ✅ Worker for {symbol} is ALIVE and RUNNING (thread ID: {worker.thread.ident if worker.thread else 'N/A'})")
                    else:
                        print(f"[MultiSymbol] ⚠️ Worker for {symbol} is NOT ALIVE (running={worker.running}, thread_alive={is_alive}, error={worker.error})")
            
            if len(active_workers) == 0:
                print("[MultiSymbol] ❌ ERROR: No active workers found after initialization!")
                self.running = False
                raise RuntimeError("No active workers found after initialization. Check logs for errors.")
            
            print(f"[MultiSymbol] 🎉 SUCCESS! MultiSymbolManager is ACTIVE with {len(active_workers)} worker(s): {', '.join(active_workers)}")
            print(f"[MultiSymbol] 📊 Manager status: running={self.running}, active_symbols={self.settings.active_symbols}, workers_count={len(active_workers)}")
            
            # Запускаем мониторинг воркеров
            try:
                self._start_monitor()
                print("[MultiSymbol] ✅ Worker monitor started successfully")
            except Exception as e:
                print(f"[MultiSymbol] ⚠️ Warning: Failed to start monitor: {e}")
                import traceback
                traceback.print_exc()
                # Мониторинг не критичен, продолжаем работу
            
            print("[MultiSymbol] ✨ MultiSymbolManager initialization COMPLETE - Bot is READY and ACTIVE! ✨")
    
    def stop(self):
        """Остановить все воркеры"""
        with self.lock:
            if not self.running:
                print("[MultiSymbol] ⚠️ Manager is not running")
                return
            
            print("[MultiSymbol] 🛑 Stopping MultiSymbolManager...")
            self.running = False
            
            # Останавливаем все воркеры
            # Примечание: run_live_from_api имеет бесконечный цикл while True
            # Нужно будет добавить механизм остановки в run_live_from_api
            # Пока просто помечаем как не запущенные
            for symbol, worker in self.workers.items():
                if worker.running:
                    print(f"[MultiSymbol] Stopping worker for {symbol}...")
                    worker.running = False
                    # Устанавливаем событие остановки
                    if worker.stop_event:
                        worker.stop_event.set()
                        print(f"[MultiSymbol] Stop event set for {symbol}")
            
            # Ждем завершения потоков (с таймаутом)
            for symbol, worker in self.workers.items():
                if worker.thread and worker.thread.is_alive():
                    print(f"[MultiSymbol] Waiting for worker thread {symbol} to stop...")
                    worker.thread.join(timeout=10.0)  # Увеличиваем таймаут до 10 секунд
                    if worker.thread.is_alive():
                        print(f"[MultiSymbol] ⚠️ Worker thread for {symbol} did not stop in time (10s timeout)")
                    else:
                        print(f"[MultiSymbol] ✅ Worker thread for {symbol} stopped successfully")
            
            # Останавливаем мониторинг
            self._stop_monitor()
            
            print("[MultiSymbol] ✅ MultiSymbolManager stopped")
    
    def add_symbol(self, symbol: str, settings: Optional[AppSettings] = None) -> bool:
        """
        Добавить новый символ в торговлю.
        
        Args:
            symbol: Торговая пара
            settings: Настройки (если None, используются общие настройки)
        
        Returns:
            True если символ добавлен успешно
        """
        with self.lock:
            available_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
            if symbol not in available_symbols:
                print(f"[MultiSymbol] ❌ Invalid symbol: {symbol}")
                return False
            
            if symbol in self.workers and self.workers[symbol].running:
                print(f"[MultiSymbol] ⚠️ Worker for {symbol} is already running")
                return False
            
            # Добавляем символ в активные
            if symbol not in self.settings.active_symbols:
                self.settings.active_symbols.append(symbol)
            
            # Создаем воркер
            worker_settings = settings or self._create_settings_for_symbol(symbol)
            self.workers[symbol] = WorkerStatus(
                symbol=symbol,
                running=False,
                settings=worker_settings
            )
            
            # Если менеджер запущен, сразу запускаем воркер
            if self.running:
                worker = self.workers[symbol]
                worker.thread = threading.Thread(
                    target=self._worker_thread,
                    args=(symbol, worker_settings),
                    name=f"BotWorker-{symbol}",
                    daemon=True
                )
                # Сохраняем время запуска потока для мониторинга
                worker.thread._start_time = time.time()
                worker.thread.start()
                print(f"[MultiSymbol] ✅ Added and started worker for {symbol}")
            else:
                print(f"[MultiSymbol] ✅ Added worker for {symbol} (will start when manager starts)")
            
            return True
    
    def remove_symbol(self, symbol: str) -> bool:
        """
        Удалить символ из торговли.
        
        Args:
            symbol: Торговая пара
        
        Returns:
            True если символ удален успешно
        """
        with self.lock:
            if symbol not in self.workers:
                print(f"[MultiSymbol] ⚠️ Worker for {symbol} not found")
                return False
            
            # Останавливаем воркер, если он запущен
            worker = self.workers[symbol]
            if worker.running:
                print(f"[MultiSymbol] Stopping worker for {symbol}...")
                worker.running = False
                if worker.thread and worker.thread.is_alive():
                    worker.thread.join(timeout=5.0)
            
            # Удаляем из активных символов
            if symbol in self.settings.active_symbols:
                self.settings.active_symbols.remove(symbol)
            
            # Удаляем воркер
            del self.workers[symbol]
            print(f"[MultiSymbol] ✅ Removed worker for {symbol}")
            return True
    
    def get_status(self) -> Dict[str, Any]:
        """
        Получить статус всех активных воркеров.
        
        Returns:
            Словарь со статусом всех воркеров
        """
        with self.lock:
            status = {
                "running": self.running,
                "active_symbols": self.settings.active_symbols,
                "workers": {}
            }
            
            for symbol, worker in self.workers.items():
                status["workers"][symbol] = {
                    "running": worker.running,
                    "error": worker.error,
                    "last_update": worker.last_update,
                    "thread_alive": worker.thread.is_alive() if worker.thread else False
                }
            
            return status
    
    def get_all_workers_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Получить статусы всех воркеров в формате {symbol: bot_state}.
        Используется для интеграции с веб-интерфейсом.
        
        Returns:
            Словарь статусов воркеров {symbol: bot_state}
        """
        with self.lock:
            # Получаем обновления статуса из run_live_from_api
            status_updates = get_worker_status_updates()
            
            workers_status = {}
            
            for symbol, worker in self.workers.items():
                # Получаем обновленный статус из _worker_status_updates
                updates = status_updates.get(symbol, {})
                
                # Используем current_status из обновлений, если он есть, иначе базовый статус
                current_status = updates.get("current_status")
                if not current_status:
                    current_status = "Running" if worker.running else "Stopped"
                
                current_phase = updates.get("current_phase")
                current_adx = updates.get("current_adx")
                
                workers_status[symbol] = {
                    "is_running": worker.running,
                    "current_status": current_status,
                    "current_phase": current_phase,
                    "current_adx": current_adx,
                    "last_action": updates.get("last_action"),
                    "last_action_time": updates.get("last_action_time"),
                    "last_signal": updates.get("last_signal"),
                    "last_signal_time": updates.get("last_signal_time"),
                    "last_error": worker.error or updates.get("error"),
                    "last_error_time": updates.get("last_error_time"),
                    "symbol": symbol
                }
            
            return workers_status
    
    def update_settings(self, new_settings: AppSettings):
        """
        Обновить настройки менеджера.
        
        Args:
            new_settings: Новые настройки
        """
        print(f"[MultiSymbol] ⚙️  update_settings() called with active_symbols: {new_settings.active_symbols if hasattr(new_settings, 'active_symbols') else 'N/A'}")
        import sys
        sys.stdout.flush()  # Принудительно выводим лог
        
        try:
            print(f"[MultiSymbol] ⚙️  About to acquire lock...")
            sys.stdout.flush()
            
            # Используем обычный with self.lock, но с дополнительным логированием
            # Если lock заблокирован, это будет ждать, но не должно зависать навсегда
            with self.lock:
                print(f"[MultiSymbol] ⚙️  Lock acquired, updating settings...")
                sys.stdout.flush()
                
                # Проверяем, изменился ли ml_model_type_for_all
                old_model_type = getattr(self.settings, 'ml_model_type_for_all', None)
                new_model_type = getattr(new_settings, 'ml_model_type_for_all', None)
                
                # Если тип модели изменился, очищаем кэш
                if old_model_type != new_model_type:
                    print(f"[MultiSymbol] 🔄 ML model type changed from {old_model_type} to {new_model_type}, clearing cache")
                    if hasattr(self, '_model_cache'):
                        self._model_cache.clear()
                    if hasattr(self, '_model_cache_keys'):
                        self._model_cache_keys.clear()
                
                self.settings = new_settings
                print(f"[MultiSymbol] ⚙️  Settings object updated")
                sys.stdout.flush()
                
                # Обновляем настройки для всех существующих воркеров
                # ОПТИМИЗАЦИЯ: Пропускаем обновление настроек для существующих воркеров, если они не изменились
                # Это значительно ускоряет update_settings() и предотвращает блокировку Flask
                print(f"[MultiSymbol] ⚙️  Skipping settings update for existing workers (will be updated when needed)...")
                sys.stdout.flush()
                
                # ВАЖНО: Не обновляем настройки существующих воркеров здесь, чтобы не блокировать Flask
                # Настройки будут обновлены при следующем обращении или при запуске воркеров
                # Это значительно ускоряет update_settings() и предотвращает блокировку
                
                # Инициализируем воркеры для новых активных символов
                print(f"[MultiSymbol] ⚙️  Initializing workers for active symbols: {self.settings.active_symbols}")
                try:
                    self._initialize_workers()
                    print(f"[MultiSymbol] ⚙️  Workers initialized, count: {len(self.workers)}")
                except Exception as e:
                    print(f"[MultiSymbol] ❌ Error initializing workers: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
                
                print(f"[MultiSymbol] ✅ Settings updated successfully, active symbols: {self.settings.active_symbols}")
                print(f"[MultiSymbol] ✅ Total workers count: {len(self.workers)}, symbols: {list(self.workers.keys())}")
        except Exception as e:
            print(f"[MultiSymbol] ❌ ERROR in update_settings(): {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _start_monitor(self):
        """Запустить фоновый поток для мониторинга воркеров."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            print("[MultiSymbol] ⚠️ Monitor thread is already running")
            return
        
        self.monitor_thread = threading.Thread(
            target=self._monitor_workers,
            name="MultiSymbolMonitor",
            daemon=True
        )
        self.monitor_thread.start()
        print(f"[MultiSymbol] 🔍 Started worker monitor (check interval: {self.monitor_interval}s)")
    
    def _stop_monitor(self):
        """Остановить мониторинг воркеров."""
        # Мониторинг остановится автоматически, когда self.running станет False
        if self.monitor_thread and self.monitor_thread.is_alive():
            print("[MultiSymbol] 🔍 Stopping worker monitor...")
    
    def _monitor_workers(self):
        """
        Фоновый поток для мониторинга состояния воркеров и автоматического рестарта упавших.
        """
        position_check_counter = 0  # Счетчик для проверки позиций (каждые 5 циклов = 2.5 минуты)
        
        while self.running:
            try:
                time.sleep(self.monitor_interval)
                
                if not self.running:
                    break
                
                # Периодически проверяем открытые позиции (каждые 5 циклов мониторинга)
                position_check_counter += 1
                if position_check_counter >= 5:  # ~2.5 минуты при monitor_interval=30s
                    position_check_counter = 0
                    try:
                        with self.lock:
                            detected = self._detect_and_add_open_positions()
                            # Если обнаружены новые символы, запускаем для них воркеры
                            for symbol in detected:
                                worker = self.workers.get(symbol)
                                if worker and not worker.running and (not worker.thread or not worker.thread.is_alive()):
                                    print(f"[MultiSymbol] 🚀 Starting worker for newly detected position: {symbol}")
                                    worker.settings = self._create_settings_for_symbol(symbol)
                                    worker.stop_event.clear()
                                    worker.thread = threading.Thread(
                                        target=self._worker_thread,
                                        args=(symbol, worker.settings),
                                        name=f"BotWorker-{symbol}",
                                        daemon=True
                                    )
                                    worker.thread._start_time = time.time()
                                    worker.thread.start()
                    except Exception as e:
                        print(f"[MultiSymbol] ⚠️ Error in periodic position check: {e}")
                
                # Получаем обновления статуса из run_live_from_api
                status_updates = get_worker_status_updates()
                
                with self.lock:
                    current_time = time.time()
                    
                    # Обновляем last_update для воркеров из статус-обновлений
                    for symbol, updates in status_updates.items():
                        worker = self.workers.get(symbol)
                        if worker and updates.get("last_update"):
                            worker.last_update = updates.get("last_update")
                            if updates.get("error"):
                                worker.error = updates.get("error")
                            else:
                                # Если ошибки нет, сбрасываем её
                                if worker.error:
                                    worker.error = None
                                    # Если воркер работает стабильно более 2 минут, сбрасываем счетчик падений
                                    if worker.last_update and (current_time - worker.last_update) < 120:
                                        worker.consecutive_failures = 0
                    
                    # Проверяем все активные воркеры
                    for symbol, worker in self.workers.items():
                        # Пропускаем воркеры, которые не должны быть запущены
                        if symbol not in self.settings.active_symbols:
                            continue
                        
                        # Проверяем, упал ли воркер
                        is_dead = False
                        reason = None
                        
                        # Проверка 1: Поток не запущен, но должен быть запущен
                        if not worker.running and symbol in self.settings.active_symbols:
                            is_dead = True
                            reason = "Worker not running"
                        
                        # Проверка 2: Поток не жив, но должен быть запущен
                        elif worker.running and (not worker.thread or not worker.thread.is_alive()):
                            is_dead = True
                            reason = "Thread is dead"
                        
                        # Проверка 3: Поток жив, но нет обновлений слишком долго (таймаут)
                        elif worker.running and worker.thread and worker.thread.is_alive():
                            if worker.last_update:
                                time_since_update = current_time - worker.last_update
                                if time_since_update > self.worker_timeout:
                                    is_dead = True
                                    reason = f"No update for {time_since_update:.0f}s (timeout: {self.worker_timeout}s)"
                            else:
                                # Если last_update вообще нет, но поток жив - возможно только что запустился
                                # Даем ему время (30 секунд)
                                if worker.thread and worker.thread.is_alive():
                                    thread_start_time = getattr(worker.thread, '_start_time', current_time)
                                    if current_time - thread_start_time > 30:
                                        is_dead = True
                                        reason = "No last_update timestamp (thread alive but no updates)"
                        
                        # Проверка 4: Есть ошибка, но воркер помечен как running
                        elif worker.running and worker.error:
                            # Если ошибка недавняя (менее минуты назад), считаем воркер упавшим
                            if worker.last_update and (current_time - worker.last_update) < 60:
                                is_dead = True
                                reason = f"Worker has error: {worker.error[:50]}"
                        
                        # Если воркер упал, пытаемся его перезапустить
                        if is_dead:
                            # Проверяем, не достигнут ли лимит рестартов, чтобы не спамить
                            if worker.restart_count < self.max_restarts:
                                print(f"[MultiSymbol] ⚠️ Detected dead worker for {symbol}: {reason}")
                                self._restart_worker(symbol, reason)
                            elif not getattr(worker, '_max_restarts_logged', False):
                                print(f"[MultiSymbol] ❌ Max restarts reached for {symbol}. Manual intervention required.")
                                worker._max_restarts_logged = True
                                worker.running = False
            
            except Exception as e:
                print(f"[MultiSymbol] ❌ Error in monitor thread: {e}")
                import traceback
                traceback.print_exc()
                # Продолжаем работу мониторинга даже при ошибке
                time.sleep(self.monitor_interval)
        
        print("[MultiSymbol] 🔍 Monitor thread stopped")
    
    def clear_model_cache(self, symbol: Optional[str] = None):
        """
        Очищает кэш моделей для указанного символа или для всех символов.
        
        Args:
            symbol: Символ для очистки кэша (если None, очищает для всех)
        """
        with self.lock:
            if symbol:
                if symbol in self._model_cache:
                    del self._model_cache[symbol]
                if symbol in self._model_cache_keys:
                    del self._model_cache_keys[symbol]
                print(f"[MultiSymbol] 🗑️  Cleared model cache for {symbol}")
            else:
                self._model_cache.clear()
                self._model_cache_keys.clear()
                print(f"[MultiSymbol] 🗑️  Cleared model cache for all symbols")
    
    def _restart_worker(self, symbol: str, reason: str = "Unknown"):
        """
        Перезапустить упавший воркер.
        
        Args:
            symbol: Торговая пара
            reason: Причина рестарта
        """
        worker = self.workers.get(symbol)
        if not worker:
            print(f"[MultiSymbol] ⚠️ Cannot restart: worker for {symbol} not found")
            return
        
        # Проверяем лимит рестартов
        if worker.restart_count >= self.max_restarts:
            print(f"[MultiSymbol] ❌ Max restarts ({self.max_restarts}) reached for {symbol}. Worker will not be restarted.")
            worker.error = f"Max restarts exceeded. Last reason: {reason}"
            worker.running = False
            return
        
        # Увеличиваем счетчик последовательных падений
        worker.consecutive_failures += 1
        
        # Если это не первое падение, добавляем экспоненциальную задержку
        restart_delay = self.restart_delay * (2 ** min(worker.consecutive_failures - 1, 3))
        
        print(f"[MultiSymbol] 🔄 Restarting worker for {symbol} (attempt {worker.restart_count + 1}/{self.max_restarts}, delay: {restart_delay:.1f}s)")
        print(f"[MultiSymbol]    Reason: {reason}")
        
        # Сигнализируем старому потоку о необходимости остановки
        if worker.stop_event:
            worker.stop_event.set()
            print(f"[MultiSymbol] Stop event set for old worker {symbol}")
        
        # Останавливаем старый поток, если он еще жив
        if worker.thread and worker.thread.is_alive():
            try:
                worker.running = False
                worker.thread.join(timeout=2.0) # Небольшой таймаут, не будем ждать долго
                if worker.thread.is_alive():
                    print(f"[MultiSymbol] ⚠️ Old thread for {symbol} still alive, it should stop soon...")
            except Exception as e:
                print(f"[MultiSymbol] ⚠️ Error stopping old thread for {symbol}: {e}")
        
        # Создаем НОВОЕ событие остановки для нового воркера
        worker.stop_event = threading.Event()
        
        # Ждем перед рестартом (в отдельном потоке, чтобы не блокировать монитор?)
        # Нет, монитор сам спит 30 секунд, но restart_delay может быть больше.
        # Для простоты пока оставим sleep здесь, но это заблокирует монитор для других символов.
        # В идеале рестарт должен быть асинхронным.
        time.sleep(restart_delay)
        
        # Проверяем, что менеджер все еще запущен
        if not self.running:
            print(f"[MultiSymbol] ⚠️ Manager stopped, aborting restart for {symbol}")
            return
        
        # Проверяем, что символ все еще активен
        if symbol not in self.settings.active_symbols:
            print(f"[MultiSymbol] ⚠️ Symbol {symbol} no longer active, aborting restart")
            return
        
        try:
            # Обновляем настройки воркера (на случай, если они изменились)
            worker.settings = self._create_settings_for_symbol(symbol)
            
            # Сбрасываем ошибку и обновляем счетчики
            worker.error = None
            worker.restart_count += 1
            worker.last_restart_time = time.time()
            
            # Создаем новый поток
            worker.thread = threading.Thread(
                target=self._worker_thread,
                args=(symbol, worker.settings),
                name=f"BotWorker-{symbol}-restart{worker.restart_count}",
                daemon=True
            )
            # Сохраняем время запуска потока для мониторинга
            worker.thread._start_time = time.time()
            worker.thread.start()
            
            print(f"[MultiSymbol] ✅ Worker for {symbol} restarted successfully (restart #{worker.restart_count})")
            
            # Если рестарт успешен, сбрасываем счетчик последовательных падений через некоторое время
            # (проверим в следующем цикле мониторинга)
        
        except Exception as e:
            print(f"[MultiSymbol] ❌ Failed to restart worker for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            worker.error = f"Restart failed: {str(e)}"
            worker.running = False
    
    def _reset_consecutive_failures(self, symbol: str):
        """
        Сбросить счетчик последовательных падений для воркера.
        Вызывается, когда воркер успешно работает некоторое время.
        
        Args:
            symbol: Торговая пара
        """
        worker = self.workers.get(symbol)
        if worker and worker.consecutive_failures > 0:
            print(f"[MultiSymbol] ✅ Worker for {symbol} is stable, resetting failure counter")
            worker.consecutive_failures = 0
