# План реализации одновременной торговли несколькими торговыми парами

## Текущая архитектура

### Что есть сейчас:
- ✅ Одна торговая пара (`settings.symbol`)
- ✅ История сделок и сигналов уже хранит `symbol` для каждой записи
- ✅ Все стратегии работают независимо от символа (используют DataFrame)
- ✅ API и клиент Bybit уже поддерживают работу с разными символами

### Что нужно изменить:
- ❌ Добавить поддержку списка активных символов в настройках
- ❌ Многопоточная/асинхронная обработка каждой пары
- ❌ Раздельное хранение состояния для каждой пары
- ❌ UI для управления и мониторинга нескольких пар одновременно

---

## Архитектурное решение

### Выбранный подход: **Threading с изоляцией контекста**

**Почему Threading, а не Multiprocessing или Async:**
- ✅ Простота реализации (минимальные изменения кода)
- ✅ Достаточно для 3-5 пар (I/O-bound операции, не CPU-intensive)
- ✅ Легкий обмен данными через shared state
- ✅ История и настройки уже в памяти
- ❌ Не multiprocessing (слишком тяжело для 3 пар)
- ❌ Не async (требует полной переделки на asyncio)

**Архитектура:**
```
Main Thread (Web Server)
│
├── Bot Manager Thread (координатор)
│   ├── Symbol Worker Thread 1 (BTCUSDT)
│   │   └── run_live_from_api(settings_btc)
│   ├── Symbol Worker Thread 2 (ETHUSDT)
│   │   └── run_live_from_api(settings_eth)
│   └── Symbol Worker Thread 3 (SOLUSDT)
│       └── run_live_from_api(settings_sol)
```

---

## Детальный план реализации

### Этап 1: Изменение структуры данных (Config)

#### 1.1. Обновить `AppSettings` в `bot/config.py`:
```python
@dataclass
class AppSettings:
    # БЫЛО: symbol: str = "BTCUSDT"
    # СТАЛО:
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT"])  # Список всех доступных пар
    active_symbols: List[str] = field(default_factory=lambda: ["BTCUSDT"])  # Активные для торговли
    primary_symbol: str = "BTCUSDT"  # Основная пара для админки (по умолчанию)
    
    # Остальное без изменений
    timeframe: str = "15m"
    # ...
```

#### 1.2. Добавить `SymbolSettings` для индивидуальных настроек:
```python
@dataclass
class SymbolSettings:
    """Настройки для конкретной торговой пары"""
    symbol: str
    enabled: bool = True  # Включена ли торговля по этой паре
    # Индивидуальные настройки риска (опционально)
    risk: Optional[RiskParams] = None  # Если None, используются общие настройки
    ml_model_path: Optional[str] = None  # Индивидуальный ML модель для пары
```

#### 1.3. Обновить `load_settings()`:
- Загружать `ACTIVE_SYMBOLS` из .env (comma-separated: "BTCUSDT,ETHUSDT,SOLUSDT")
- Загружать `PRIMARY_SYMBOL` для UI
- Валидация символов (только допустимые пары)

---

### Этап 2: Менеджер ботов (Bot Manager)

#### 2.1. Создать `bot/multi_symbol_manager.py`:

```python
class MultiSymbolManager:
    """Управляет несколькими торговыми ботами одновременно"""
    
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.workers: Dict[str, Thread] = {}  # {symbol: thread}
        self.worker_settings: Dict[str, AppSettings] = {}  # {symbol: settings}
        self.running = False
        self.lock = threading.Lock()
    
    def start(self):
        """Запустить торговлю для всех активных символов"""
        
    def stop(self):
        """Остановить все воркеры"""
        
    def add_symbol(self, symbol: str, settings: AppSettings):
        """Добавить новый символ в торговлю"""
        
    def remove_symbol(self, symbol: str):
        """Удалить символ из торговли"""
        
    def get_status(self) -> Dict[str, Any]:
        """Получить статус всех активных воркеров"""
```

#### 2.2. Модифицировать `run_live_from_api()`:
- Добавить параметр `symbol: str` (вместо `settings.symbol`)
- Создать локальную копию `settings` с переопределенным `symbol`
- Все обращения к `settings.symbol` заменить на параметр `symbol`
- Использовать `symbol` как ключ для словарей состояния (position_max_profit, etc.)

---

### Этап 3: Изоляция состояния (State Isolation)

#### 3.1. Обновить словари состояния в `run_live_from_api()`:
```python
# БЫЛО (глобальные для всех символов):
position_max_profit: Dict[str, float] = {}  # {symbol: max_profit}
position_max_price: Dict[str, float] = {}   # {symbol: max_price}

# СТАНЕТ (локальные для каждого воркера, но с ключом symbol):
# Внутри каждой функции использовать symbol как ключ
position_max_profit = {}  # {symbol: max_profit} - локальный для воркера
```

#### 3.2. Обработка processed_signals:
```python
# Хранить отдельно для каждой пары
processed_signals_file = f"processed_signals_{symbol}.json"
```

#### 3.3. Обновить `_save_processed_signals()`:
- Принимать `symbol` как параметр
- Использовать файл `processed_signals_{symbol}.json`

---

### Этап 4: Веб-интерфейс (UI)

#### 4.1. Настройка активных символов:
**Новая вкладка "Symbols" в админке:**
- Список всех доступных символов (BTCUSDT, ETHUSDT, SOLUSDT)
- Чекбоксы для включения/выключения каждой пары
- Выбор основной пары (для отображения графика по умолчанию)
- Сохранение в .env и перезапуск воркеров

#### 4.2. Dashboard:
- **Общий обзор**: Сводная статистика по всем активным парам
  - Общий баланс
  - Общий PnL (сегодня, неделя, месяц)
  - Количество открытых позиций
  - Список всех открытых позиций с символами

#### 4.3. Переключение между парами:
- **Выпадающий список** "Выбрать пару" в верхней части админки
- При переключении обновляется:
  - График (показывает выбранную пару)
  - История сделок (фильтр по символу)
  - История сигналов (фильтр по символу)
  - Текущие позиции/ордера (фильтр по символу)

#### 4.4. Таблицы с фильтрацией:
- **Trade History**: Добавить фильтр по символу + общий вид
- **Signal History**: Добавить фильтр по символу + общий вид
- Колонка "Symbol" во всех таблицах

#### 4.5. График:
- При загрузке данных для графика использовать выбранный `primary_symbol`
- Кнопка переключения символа рядом с графиком

---

### Этап 5: API Endpoints

#### 5.1. Новые endpoints в `bot/web/app.py`:

```python
@app.route("/api/symbols/list")
def api_symbols_list():
    """Получить список всех доступных символов"""
    
@app.route("/api/symbols/active")
def api_symbols_active():
    """Получить список активных символов"""
    
@app.route("/api/symbols/set-active", methods=["POST"])
def api_symbols_set_active():
    """Установить активные символы"""
    # data: {"symbols": ["BTCUSDT", "ETHUSDT"], "primary": "BTCUSDT"}
    
@app.route("/api/status/all")
def api_status_all():
    """Получить статус для всех активных символов"""
    # Возвращает: {symbol: {balance, position, orders, ...}}
    
@app.route("/api/chart/data/<symbol>")
def api_chart_data_symbol(symbol):
    """Получить данные графика для конкретного символа"""
```

#### 5.2. Обновить существующие endpoints:
- Добавить параметр `symbol` (опциональный) для фильтрации
- `/api/trades` - фильтр по символу
- `/api/signals` - фильтр по символу

---

### Этап 6: Интеграция с существующим кодом

#### 6.1. Модификации в `bot/live.py`:

**Изменения в `run_live_from_api()`:**
```python
def run_live_from_api(
    initial_settings: AppSettings,
    symbol: str,  # НОВЫЙ ПАРАМЕТР
    bot_state: Optional[Dict[str, Any]] = None,
    signal_max_age_seconds: int = 60,
) -> None:
    # Создаем локальную копию настроек с переопределенным символом
    settings = initial_settings
    settings.symbol = symbol  # Переопределяем
    
    # Все обращения к settings.symbol теперь используют локальный symbol
    # ...
```

**Замена всех `settings.symbol` на `symbol` внутри функции:**
- `_get_position(client, symbol)`
- `_get_open_orders(client, symbol)`
- `client.place_order(symbol=symbol, ...)`
- `position_strategy[symbol]`
- `position_max_profit[symbol]`
- и т.д.

#### 6.2. Обновить `main.py`:
```python
# В режиме web запускать MultiSymbolManager вместо run_live_from_api
if args.mode == "web":
    from bot.web.app import run_web_server
    # run_web_server запускает MultiSymbolManager внутри
    run_web_server(...)
```

#### 6.3. Обновить `bot/web/app.py`:
```python
# Глобальный менеджер
multi_symbol_manager: Optional[MultiSymbolManager] = None

def init_app():
    global multi_symbol_manager
    settings = load_settings()
    # Инициализация менеджера
    multi_symbol_manager = MultiSymbolManager(settings)
    if settings.active_symbols:
        multi_symbol_manager.start()

@app.route("/api/bot/start", methods=["POST"])
def api_bot_start():
    """Запустить бота для всех активных символов"""
    if multi_symbol_manager:
        multi_symbol_manager.start()
    return jsonify({"success": True})

@app.route("/api/bot/stop", methods=["POST"])
def api_bot_stop():
    """Остановить все воркеры"""
    if multi_symbol_manager:
        multi_symbol_manager.stop()
    return jsonify({"success": True})
```

---

## Этапы реализации (приоритет)

### Фаза 1: Базовая инфраструктура (1-2 дня)
1. ✅ Изменение `AppSettings` и `load_settings()`
2. ✅ Создание `MultiSymbolManager` с базовой логикой
3. ✅ Модификация `run_live_from_api()` для работы с `symbol` параметром
4. ✅ Изоляция состояния (processed_signals, position tracking)

### Фаза 2: UI для управления символами (1 день)
5. ✅ Вкладка "Symbols" в админке
6. ✅ API endpoints для управления символами
7. ✅ Сохранение в .env и перезагрузка

### Фаза 3: Обновление интерфейса (1-2 дня)
8. ✅ Переключатель символов в верхней части
9. ✅ Фильтрация таблиц по символу
10. ✅ Обновление графика для выбранного символа
11. ✅ Общий dashboard с сводной статистикой

### Фаза 4: Тестирование и оптимизация (1 день)
12. ✅ Тестирование на тестнете с 2-3 парами
13. ✅ Проверка изоляции состояния
14. ✅ Проверка производительности
15. ✅ Исправление багов

---

## Важные моменты и риски

### ⚠️ Потенциальные проблемы:

1. **Race Conditions:**
   - Использовать `threading.Lock()` для синхронизации доступа к shared state
   - Изоляция состояния на уровне воркера

2. **Rate Limits Bybit:**
   - Каждая пара делает свои запросы
   - Нужно контролировать общее количество запросов
   - Возможно, добавить задержки между запросами разных воркеров

3. **Баланс и лимиты:**
   - Учитывать общий баланс при расчете размера позиций
   - `balance_percent_per_trade` должен считаться от общего баланса, но распределяться между парами

4. **Логирование:**
   - Добавить префикс `[symbol]` во все логи
   - Пример: `[BTCUSDT] ✅ Position opened` вместо `[live] ✅ Position opened`

5. **Обработка ошибок:**
   - Если один воркер упал, остальные должны продолжать работать
   - Автоматический рестарт упавших воркеров

6. **Memory usage:**
   - Каждый воркер держит свои данные в памяти
   - Для 3 пар это не критично, но нужно следить

---

## Примеры использования

### .env файл:
```env
# Доступные символы (через запятую)
AVAILABLE_SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT

# Активные символы для торговли
ACTIVE_SYMBOLS=BTCUSDT,ETHUSDT

# Основной символ для UI
PRIMARY_SYMBOL=BTCUSDT

# Остальные настройки остаются общими
TIMEFRAME=15m
# ...
```

### Использование в коде:
```python
# Запуск менеджера
settings = load_settings()
manager = MultiSymbolManager(settings)
manager.start()  # Запускает торговлю для ACTIVE_SYMBOLS

# Добавление нового символа
manager.add_symbol("SOLUSDT", settings)

# Удаление символа
manager.remove_symbol("ETHUSDT")

# Получение статуса
status = manager.get_status()
# {symbol: {running: True, last_signal: ..., position: ...}}
```

---

## Альтернативные подходы (не выбраны, но можно рассмотреть)

### Вариант A: AsyncIO (asyncio)
- ✅ Более эффективное использование ресурсов
- ❌ Требует полной переделки на async/await
- ❌ Сложность отладки
- ❌ Все библиотеки должны поддерживать async

### Вариант B: Multiprocessing
- ✅ Полная изоляция процессов
- ❌ Сложность обмена данными
- ❌ Больше потребление памяти
- ❌ Сложность управления

### Вариант C: Очередь задач (Single Thread)
- ✅ Простота реализации
- ❌ Не настоящий параллелизм
- ❌ Одна пара блокирует другие

---

## Вопросы для обсуждения

1. **Стратегии:** Одинаковые для всех пар или индивидуальные?
   - Рекомендация: Общие настройки по умолчанию, возможность индивидуальной настройки в будущем

2. **Риск-менеджмент:** Общий баланс или раздельный?
   - Рекомендация: Общий баланс, но с учетом всех открытых позиций

3. **ML модели:** Одна модель на пару или общая?
   - Рекомендация: Индивидуальная модель для каждой пары (уже есть `xgb_BTCUSDT_15.pkl`, `xgb_ETHUSDT_15.pkl`)

4. **UI:** Один график с переключением или несколько одновременно?
   - Рекомендация: Один график с переключением (проще для мобильных)

---

## Итоговая структура файлов

```
bot/
├── config.py              # Обновлен: symbols, active_symbols
├── live.py                # Обновлен: параметр symbol
├── multi_symbol_manager.py # НОВЫЙ: менеджер воркеров
└── web/
    ├── app.py             # Обновлен: MultiSymbolManager, новые endpoints
    ├── history.py         # Без изменений (уже поддерживает symbol)
    └── templates/
        └── index.html     # Обновлен: UI для управления символами
```

---

## Следующие шаги

1. ✅ Обсудить план с пользователем
2. ⬜ Начать реализацию с Фазы 1
3. ⬜ Тестирование на каждой фазе
4. ⬜ Постепенный деплой
