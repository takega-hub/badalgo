import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from crypto_env import CryptoTradingEnv
from data_processor import DataProcessor

def run_backtest():
    # 1. Подготовка данных
    processor = DataProcessor("data/btc_15m.csv")
    processor.load_data()
    df = processor.prepare_features()
    _, test_df = processor.split_data(test_size=0.2)
    obs_cols = processor.get_observation_columns()

    # 2. Инициализация среды (те же параметры, что при обучении!)
    env = CryptoTradingEnv(test_df, obs_cols, rr_ratio=2.5, atr_multiplier=2.0)
    
    # 3. Загрузка модели
    model = PPO.load("ppo_crypto_bot_v4_1")

    # 4. Цикл теста
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)

    # 5. Визуализация результатов из лога
    log_df = pd.read_csv("trades_log.csv").tail(100) # Посмотрим последние 100 сделок
    plt.figure(figsize=(15, 7))
    plt.plot(log_df['net_worth'].values, label='Equity Curve')
    plt.title("Backtest Equity - v4.1")
    plt.legend()
    plt.show()
    
    print(f"Финальный баланс на тесте: {env.net_worth:.2f}")

if __name__ == "__main__":
    run_backtest()