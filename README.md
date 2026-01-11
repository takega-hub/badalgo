# Crypto Bot (Bybit Futures, Python)

Каркас торгового бота под Bybit Futures c возможностью симуляции (paper trading) и стратегией с фильтром тренда по ADX/DI на 4H и точками входа/добавления на 15m.

## Быстрый старт
1. Активируй виртуальное окружение (если не активно) и установи зависимости:
   ```
   pip install -r requirements.txt
   ```
2. Создай `.env` (по примеру в `config/settings.py`) с API ключами Bybit, если планируешь реальную торговлю.
3. Подготовь 15m исторические свечи в CSV с колонками: `timestamp,open,high,low,close,volume` (timestamp в ISO или миллисекундах). Для бэктеста см. `main.py`.
4. Запусти симуляцию:
   ```
   python main.py --mode backtest --data data/your_15m.csv
   ```

## Стратегия (сводка)
- **4H фильтр:** bias активен, если ADX(4H) > 25. LONG, если PlusDI > MinusDI; SHORT, если MinusDI > PlusDI; иначе RANGE.
- **15m входы:** 
  - Брейкаут/«fakeout recovery» через пробой недавнего экстремума (по умолчанию lookback=20) с объёмом > средний.
  - Скейлинг (усреднение): добор на откате к SMA20(15m) при условии volume spike.
  - Скейлинг (пирамидинг): добор на пробое консолидации, если объём > средний и RSI не в экстремумах.
- Все параметры регулируются в `bot/config.py`.

## Режимы работы

### Backtest
```bash
python main.py --mode backtest --data data/your_15m.csv
```

### Backtest from API
```bash
python main.py --mode backtest_api --limit 500
```

### Live Trading
```bash
python main.py --mode live_api
```

### Manual Control
```bash
# Просмотр статуса
python main.py --mode manual --action status

# Открытие позиции
python main.py --mode manual --action open --side long --usd 50

# Закрытие позиции
python main.py --mode manual --action close
```

### Web Admin Panel
```bash
# Запуск веб-админки на http://127.0.0.1:5000
python main.py --mode web

# Или с кастомным портом и хостом (для доступа с других машин)
python main.py --mode web --web-host 0.0.0.0 --web-port 5000
```

Веб-админка предоставляет:
- **Dashboard**: Статус бота, баланс, позиции, открытые ордера, интерактивный график с индикаторами
- **Strategies**: Выбор активных стратегий (Trend/Flat/ML) и приоритет при конфликте сигналов
- **Strategy Settings**: Настройка параметров всех стратегий (ADX, DI, SMA, RSI, BB, и т.д.)
- **Risk Settings**: Управление размерами ордеров, TP/SL, плечом, защитами
- **Trades**: История всех сделок с PnL (сегодня, неделя, месяц, всего), статистика по паре
- **Signals**: История всех сигналов бота с фильтрацией и цветовой визуализацией
- **Settings**: Управление API ключами Bybit, смена пароля администратора
- **Мобильная версия**: Адаптивный интерфейс с автоматическим определением устройства и ручным переключением режимов

## Деплой на сервер

Полная инструкция по деплою на сервер находится в файле [DEPLOY.md](DEPLOY.md).

### Быстрый деплой:

1. **Подготовка сервера:**
   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo apt install -y python3 python3-pip python3-venv git
   ```

2. **Клонирование и установка:**
   ```bash
   cd /opt
   sudo git clone <your-repo> crypto_bot
   sudo chown -R $USER:$USER crypto_bot
   cd crypto_bot
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Настройка:**
   ```bash
   cp env.example .env
   nano .env  # Заполните все переменные
   ```

4. **Запуск через systemd:**
   ```bash
   sudo cp crypto-bot.service /etc/systemd/system/
   sudo nano /etc/systemd/system/crypto-bot.service  # Измените пути и пользователя
   sudo systemctl daemon-reload
   sudo systemctl enable crypto-bot
   sudo systemctl start crypto-bot
   sudo systemctl status crypto-bot
   ```

5. **Проверка:**
   - Откройте браузер: `http://your-server-ip:5000`
   - Войдите с учетными данными из `.env`

### Использование скрипта запуска:

```bash
chmod +x start.sh
./start.sh --host 0.0.0.0 --port 5000
```

## Файлы проекта
- `main.py` — CLI-запуск симуляции/live-бота/веб-админки.
- `bot/strategy.py` — логика сигналов (трендовая и флэтовая стратегии).
- `bot/indicators.py` — расчёт индикаторов и агрегация 15m→4H/1H.
- `bot/simulation.py` — простой paper trading на исторических данных.
- `bot/exchange/bybit_client.py` — обёртка над Bybit API.
- `bot/config.py` — параметры стратегии, рисков и API.
- `bot/live.py` — логика live-торговли с управлением TP/SL, позициями.
- `bot/web/app.py` — Flask веб-приложение для админки.
- `bot/web/history.py` — хранение истории сделок и сигналов.
- `bot/ml/` — ML стратегия и обучение моделей.
- `DEPLOY.md` — подробная инструкция по деплою на сервер.
- `crypto-bot.service` — systemd service файл для запуска как службы.
- `start.sh` — скрипт для запуска бота.
- `env.example` — пример конфигурации переменных окружения.

