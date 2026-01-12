# Настройка тестового сервера Bybit (Testnet)

## Шаг 1: Создайте файл `.env` в корне проекта

Скопируйте содержимое из `env.example` и обновите следующие параметры:

```env
# Bybit Testnet API Settings
BYBIT_API_KEY=Oqe7oPIvtBs60iIoIz
BYBIT_API_SECRET=BmJvjEknyMgq8MWGybVPG8OopxUy2qzFaxdc
BYBIT_BASE_URL=https://api-testnet.bybit.com
```

## Шаг 2: Настройте стратегии для тестирования

В файле `.env` вы можете включить новые стратегии:

```env
# Включить новую стратегию импульсного пробоя (вместо старой трендовой)
ENABLE_MOMENTUM_STRATEGY=true

# Включить новую стратегию возврата к среднему (вместо старой флэтовой)
ENABLE_MEAN_REV_STRATEGY=true

# Или использовать старые стратегии
ENABLE_TREND_STRATEGY=true
ENABLE_FLAT_STRATEGY=true
```

## Шаг 3: Запустите бота

```bash
# Активируйте виртуальное окружение
venv\Scripts\activate  # Windows
# или
source venv/bin/activate  # Linux/Mac

# Запустите бота в режиме веб-интерфейса
python main.py --mode web --web-host 0.0.0.0 --web-port 5000
```

## Шаг 4: Проверьте подключение

1. Откройте браузер и перейдите на `http://localhost:5000`
2. Войдите с учетными данными из `.env` (ADMIN_USERNAME и ADMIN_PASSWORD)
3. Проверьте статус бота и подключение к тестовому серверу

## Важные замечания

- **Testnet использует тестовые средства** - это безопасно для тестирования
- **API ключи для testnet отличаются от production** - используйте только тестовые ключи
- **Данные на testnet могут отличаться от production** - это нормально
- **Все сделки на testnet виртуальные** - реальные деньги не используются

## Параметры новых стратегий

### Momentum Breakout Strategy (Импульсный пробой)

```env
EMA_FAST_LENGTH=20
EMA_SLOW_LENGTH=50
MOMENTUM_ADX_THRESHOLD=25.0
MOMENTUM_VOLUME_SPIKE_MIN=1.5
MOMENTUM_VOLUME_SPIKE_MAX=2.0
MOMENTUM_TRAILING_STOP_EMA=true
MOMENTUM_EMA_TIMEFRAME=1h
```

### Mean Reversion Strategy (Возврат к среднему)

```env
MEAN_REV_RSI_OVERSOLD=30.0
MEAN_REV_VOLUME_EXHAUSTION_MULT=1.0
MEAN_REV_TP_BB_MIDDLE=true
MEAN_REV_STOP_LOSS_PCT=0.01
```

## После тестирования

Когда вы будете готовы развернуть на production сервере:

1. Обновите `.env` на сервере с production API ключами
2. Измените `BYBIT_BASE_URL=https://api.bybit.com`
3. Запустите обновление через `update.sh` на сервере
