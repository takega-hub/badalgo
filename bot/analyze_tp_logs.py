import pandas as pd
import numpy as np
import os
from datetime import datetime

class TPLogAnalyzer:
    def __init__(self, log_file="logs\v16_active\train_log.csv"):
        self.log_file = log_file
        
    def analyze_with_tp_correction(self):
        """Анализ логов с учетом частичных TP"""
        if not os.path.exists(self.log_file):
            print(f"❌ Файл не найден: {self.log_file}")
            return
        
        try:
            df = pd.read_csv(self.log_file)
            print(f"\n📊 АНАЛИЗ ЛОГОВ: {self.log_file}")
            print(f"   Строк: {len(df):,}")
            print(f"   Колонок: {len(df.columns)}")
            
            if len(df) == 0:
                print("❌ Файл пустой")
                return
            
            # 1. Исправляем TP статистику
            print("\n" + "="*60)
            print("🎯 ИСПРАВЛЕННАЯ СТАТИСТИКА TP")
            print("="*60)
            
            # Способ 1: По колонке tp_closed
            if 'tp_closed' in df.columns:
                tp_trades = df[df['tp_closed'] != 'NONE']
                print(f"   Сделок с TP (по tp_closed): {len(tp_trades)}/{len(df)} ({len(tp_trades)/len(df)*100:.1f}%)")
            
            # Способ 2: По partial_closes
            if 'partial_closes' in df.columns:
                partial_tp_trades = df[df['partial_closes'] > 0]
                print(f"   Сделок с частичными TP: {len(partial_tp_trades)}/{len(df)} ({len(partial_tp_trades)/len(df)*100:.1f}%)")
            
            # Способ 3: Анализируем exit_reason
            if 'exit_reason' in df.columns:
                print(f"\n   📋 СТАТИСТИКА ВЫХОДОВ:")
                exit_stats = df['exit_reason'].value_counts()
                for reason, count in exit_stats.items():
                    percentage = count / len(df) * 100
                    print(f"      {reason}: {count} ({percentage:.1f}%)")
                
                # Ищем скрытые TP в комментариях
                tp_keywords = ['TP', 'TAKE', 'PROFIT']
                hidden_tp = 0
                for reason in df['exit_reason']:
                    if any(keyword in str(reason).upper() for keyword in tp_keywords):
                        hidden_tp += 1
                
                if hidden_tp > 0:
                    print(f"      Скрытых TP (по ключевым словам): {hidden_tp} ({hidden_tp/len(df)*100:.1f}%)")
            
            # 2. Детальный анализ последних сделок
            print("\n" + "="*60)
            print("🔍 ДЕТАЛЬНЫЙ АНАЛИЗ ПОСЛЕДНИХ СДЕЛОК")
            print("="*60)
            
            # Показываем последние 10 сделок с деталями
            last_trades = df.tail(10)
            for idx, row in last_trades.iterrows():
                print(f"\n   #{row.get('step', idx)} {row.get('type', 'N/A')}:")
                print(f"      Entry: {row.get('entry', 'N/A'):.2f}, Exit: {row.get('exit', 'N/A'):.2f}")
                print(f"      PnL: {row.get('pnl_percent', 'N/A')}")
                print(f"      Причина: {row.get('exit_reason', 'N/A')}")
                print(f"      Длительность: {row.get('duration', 'N/A')} шагов")
                
                # Информация о TP
                if 'tp_closed' in row and row['tp_closed'] != 'NONE':
                    print(f"      TP уровни закрыты: {row['tp_closed']}")
                if 'partial_closes' in row:
                    print(f"      Частичных закрытий: {row['partial_closes']}")
            
            # 3. Анализ эффективности TP
            print("\n" + "="*60)
            print("📈 ЭФФЕКТИВНОСТЬ TP СТРАТЕГИИ")
            print("="*60)
            
            if 'pnl_percent' in df.columns:
                # Извлекаем числовые значения PnL
                pnl_values = []
                for val in df['pnl_percent']:
                    try:
                        if isinstance(val, str):
                            num = float(val.replace('%', '').replace(' ', ''))
                        else:
                            num = float(val)
                        pnl_values.append(num)
                    except:
                        pnl_values.append(0)
                
                if pnl_values:
                    # Разделяем по типам выходов
                    if 'exit_reason' in df.columns:
                        unique_reasons = df['exit_reason'].unique()
                        for reason in unique_reasons:
                            mask = df['exit_reason'] == reason
                            if mask.any():
                                reason_pnl = [pnl_values[i] for i in range(len(pnl_values)) if mask.iloc[i]]
                                if reason_pnl:
                                    avg_pnl = np.mean(reason_pnl)
                                    win_rate = sum(1 for x in reason_pnl if x > 0) / len(reason_pnl) * 100
                                    print(f"   {reason}:")
                                    print(f"      Сделок: {len(reason_pnl)}")
                                    print(f"      Средний PnL: {avg_pnl:.2f}%")
                                    print(f"      Win Rate: {win_rate:.1f}%")
                    
                    # Общая статистика
                    print(f"\n   📊 ОБЩАЯ СТАТИСТИКА:")
                    print(f"      Средний PnL: {np.mean(pnl_values):.2f}%")
                    print(f"      Win Rate: {sum(1 for x in pnl_values if x > 0)/len(pnl_values)*100:.1f}%")
                    print(f"      Лучшая сделка: {max(pnl_values):.2f}%")
                    print(f"      Худшая сделка: {min(pnl_values):.2f}%")
            
            # 4. Сохраняем исправленный анализ
            self.save_analysis_report(df)
            
        except Exception as e:
            print(f"❌ Ошибка анализа: {e}")
            import traceback
            traceback.print_exc()
    
    def save_analysis_report(self, df):
        """Сохранение отчета анализа"""
        report_file = f"tp_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Создаем улучшенный DataFrame
        report_data = []
        
        for idx, row in df.iterrows():
            # Определяем тип выхода
            exit_type = str(row.get('exit_reason', 'UNKNOWN'))
            is_tp = False
            
            # Проверяем разные признаки TP
            if 'tp_closed' in row and row['tp_closed'] != 'NONE':
                is_tp = True
            elif 'partial_closes' in row and row['partial_closes'] > 0:
                is_tp = True
            elif any(keyword in exit_type.upper() for keyword in ['TP', 'TAKE', 'PROFIT']):
                is_tp = True
            
            # Извлекаем PnL
            pnl_str = str(row.get('pnl_percent', '0'))
            try:
                pnl_value = float(pnl_str.replace('%', '').replace(' ', ''))
            except:
                pnl_value = 0
            
            report_data.append({
                'step': row.get('step', idx),
                'type': row.get('type', 'UNKNOWN'),
                'entry': row.get('entry', 0),
                'exit': row.get('exit', 0),
                'pnl_percent': pnl_value,
                'pnl_str': pnl_str,
                'exit_reason': exit_type,
                'is_tp': is_tp,
                'tp_levels_closed': row.get('tp_closed', 'NONE'),
                'partial_closes_count': row.get('partial_closes', 0),
                'duration': row.get('duration', 0),
                'trailing': row.get('trailing', 'NO'),
                'net_worth': row.get('net_worth', 0)
            })
        
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(report_file, index=False)
        
        print(f"\n✅ Отчет сохранен: {report_file}")
        
        # Выводим итоговую статистику
        total_trades = len(report_df)
        tp_trades = sum(report_df['is_tp'])
        profitable_trades = sum(1 for x in report_df['pnl_percent'] if x > 0)
        
        print(f"\n📋 ИТОГОВАЯ СТАТИСТИКА:")
        print(f"   Всего сделок: {total_trades}")
        print(f"   TP сделок (исправлено): {tp_trades} ({tp_trades/total_trades*100:.1f}%)")
        print(f"   Прибыльных сделок: {profitable_trades} ({profitable_trades/total_trades*100:.1f}%)")
        print(f"   Средний PnL: {report_df['pnl_percent'].mean():.2f}%")
        
        if tp_trades > 0:
            tp_pnl = report_df[report_df['is_tp']]['pnl_percent'].mean()
            print(f"   Средний PnL TP сделок: {tp_pnl:.2f}%")
        
        # Анализ по уровням TP
        if 'tp_levels_closed' in report_df.columns:
            print(f"\n🎯 АНАЛИЗ ПО УРОВНЯМ TP:")
            for i in range(1, 4):
                level_trades = report_df[report_df['tp_levels_closed'].astype(str).str.contains(str(i))]
                if len(level_trades) > 0:
                    print(f"   TP{i}: {len(level_trades)} сделок, средний PnL: {level_trades['pnl_percent'].mean():.2f}%")


def main():
    """Главная функция"""
    print("\n" + "="*60)
    print("🎯 АНАЛИЗ TP СДЕЛОК С ИСПРАВЛЕНИЯМИ")
    print("="*60)
    
    # Проверяем доступные файлы логов
    log_files = []
    for file in os.listdir("."):
        if file.endswith('.csv') and ('trade' in file.lower() or 'log' in file.lower()):
            log_files.append(file)
    
    for file in ['logs/v16_historical_btc/', './logs/']:
        if os.path.exists(file):
            for log_file in os.listdir(file):
                if log_file.endswith('.csv'):
                    log_files.append(os.path.join(file, log_file))
    
    if not log_files:
        print("❌ Не найдено файлов логов")
        return
    
    print(f"📂 Найдено файлов логов: {len(log_files)}")
    
    for i, log_file in enumerate(log_files, 1):
        print(f"\n{'='*40}")
        print(f"📊 ФАЙЛ {i}: {log_file}")
        print(f"{'='*40}")
        
        analyzer = TPLogAnalyzer(log_file)
        analyzer.analyze_with_tp_correction()


if __name__ == "__main__":
    main()