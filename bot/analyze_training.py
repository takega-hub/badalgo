import pandas as pd
import numpy as np

def analyze_current_log():
    try:
        df = pd.read_csv('v16_rr2_train_log.csv')
        
        print(f"\n=== АНАЛИЗ {len(df)} СДЕЛОК ===")
        
        if len(df) > 0:
            # Конвертируем проценты в числа
            df['pnl_numeric'] = df['pnl_percent'].str.replace('%','').astype(float)
            
            # Основные метрики
            win_rate = (df['pnl_numeric'] > 0).mean()
            avg_win = df[df['pnl_numeric'] > 0]['pnl_numeric'].mean()
            avg_loss = df[df['pnl_numeric'] < 0]['pnl_numeric'].mean()
            
            print(f"Win Rate: {win_rate:.1%}")
            print(f"Средняя прибыль: {avg_win:.2f}%")
            print(f"Средний убыток: {avg_loss:.2f}%")
            
            if avg_loss < 0:
                profit_factor = abs(avg_win / avg_loss)
                print(f"Profit Factor: {profit_factor:.2f}")
            
            # Анализ по типам выхода
            print(f"\nТипы выхода:")
            for exit_type in df['exit_reason'].unique():
                count = (df['exit_reason'] == exit_type).sum()
                avg_pnl = df[df['exit_reason'] == exit_type]['pnl_numeric'].mean()
                print(f"  {exit_type}: {count} сделок, средний PnL: {avg_pnl:.2f}%")
            
            # График кумулятивного PnL
            import matplotlib.pyplot as plt
            df['cumulative_pnl'] = df['pnl_numeric'].cumsum()
            df['cumulative_pnl'].plot(title='Кумулятивный PnL')
            plt.xlabel('Сделка')
            plt.ylabel('PnL (%)')
            plt.grid(True)
            plt.show()
        else:
            print("Нет сделок в логах. Агент, вероятно, всегда выбирает HOLD.")
            
    except FileNotFoundError:
        print("Файл логов не найден")
    except Exception as e:
        print(f"Ошибка анализа: {e}")

if __name__ == "__main__":
    analyze_current_log()