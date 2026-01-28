"""
Детальный анализ логов обучения V18 MTF
Выявляет проблемы и дает рекомендации по улучшению MTF стратегии
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def analyze_training_logs(log_file: str = None):
    """Детальный анализ логов обучения V18 MTF"""
    
    # Определяем путь к логам V18 MTF
    if log_file is None:
        possible_paths = [
            './logs/v18_mtf/train_v18_mtf_log.csv',
            '../logs/v18_mtf/train_v18_mtf_log.csv',
            'logs/v18_mtf/train_v18_mtf_log.csv',
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                log_file = path
                break
        
        if log_file is None:
            print("❌ Файл логов V18 MTF не найден!")
            print("Проверьте пути:")
            for path in possible_paths:
                print(f"  - {os.path.abspath(path)}")
            return
    
    if not os.path.exists(log_file):
        print(f"❌ Файл {log_file} не найден!")
        print(f"   Проверьте путь: {os.path.abspath(log_file)}")
        return
    
    print("="*70)
    print("📊 ДЕТАЛЬНЫЙ АНАЛИЗ ЛОГОВ ОБУЧЕНИЯ V18 MTF")
    print("="*70)
    
    # Загрузка данных
    try:
        # Пробуем прочитать с заголовками
        df = pd.read_csv(log_file)
        
        # Проверяем, есть ли заголовки или данные начинаются сразу
        if len(df.columns) == 1 or df.columns[0] == 'step':
            # Данные без заголовков, читаем заново с указанием колонок
            column_names = [
                'step', 'type', 'entry', 'sl_initial', 'sl_current',
                'tp_levels', 'exit', 'pnl_percent', 'net_worth',
                'exit_reason', 'duration', 'trailing', 'tp_closed', 'partial_closes',
                'trade_quality', 'rr_ratio'
            ]
            df = pd.read_csv(log_file, names=column_names, header=None)
            
            # Пропускаем первую строку если она содержит заголовки
            if len(df) > 0:
                first_val = str(df.iloc[0]['step'])
                if first_val == 'step' or first_val.startswith('step'):
                    df = df.iloc[1:].copy()
        elif 'pnl_percent' not in df.columns:
            # Пробуем определить колонки по позиции
            if len(df.columns) >= 8:
                column_names = [
                    'step', 'type', 'entry', 'sl_initial', 'sl_current',
                    'tp_levels', 'exit', 'pnl_percent', 'net_worth',
                    'exit_reason', 'duration', 'trailing', 'tp_closed', 'partial_closes',
                    'trade_quality', 'rr_ratio'
                ]
                df.columns = column_names[:len(df.columns)]
        
        print(f"\n✅ Загружено {len(df)} сделок из {log_file}")
    except Exception as e:
        print(f"❌ Ошибка загрузки файла: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if len(df) == 0:
        print("❌ Файл пуст!")
        return
    
    # Конвертируем pnl_percent в числовой формат
    def parse_pnl(pnl_str):
        try:
            if pd.isna(pnl_str):
                return 0.0
            if isinstance(pnl_str, str):
                cleaned = pnl_str.replace('%', '').replace(' ', '').strip()
                return float(cleaned)
            return float(pnl_str)
        except:
            return 0.0
    
    df['pnl_value'] = df['pnl_percent'].apply(parse_pnl)
    
    # Удаляем строки с NaN в pnl_value
    df = df[df['pnl_value'].notna()].copy()
    
    if len(df) == 0:
        print("❌ Нет валидных данных после обработки!")
        return
    
    # ==================== БАЗОВАЯ СТАТИСТИКА ====================
    print("\n" + "="*70)
    print("📈 БАЗОВАЯ СТАТИСТИКА")
    print("="*70)
    
    total_trades = len(df)
    profitable = len(df[df['pnl_value'] > 0])
    losses = len(df[df['pnl_value'] < 0])
    breakeven = len(df[df['pnl_value'] == 0])
    
    win_rate = profitable / total_trades * 100 if total_trades > 0 else 0
    avg_pnl = df['pnl_value'].mean()
    total_pnl = df['pnl_value'].sum()
    
    print(f"\nВсего сделок: {total_trades}")
    print(f"  ✅ Прибыльных: {profitable} ({profitable/total_trades*100:.1f}%)")
    print(f"  ❌ Убыточных: {losses} ({losses/total_trades*100:.1f}%)")
    print(f"  ⚖️  Безубыточных: {breakeven} ({breakeven/total_trades*100:.1f}%)")
    print(f"\nWin Rate: {win_rate:.1f}%")
    print(f"Средний PnL: {avg_pnl:.3f}%")
    print(f"Общий PnL: {total_pnl:.2f}%")
    
    # ==================== РАСПРЕДЕЛЕНИЕ LONG/SHORT ====================
    print("\n" + "="*70)
    print("🔄 РАСПРЕДЕЛЕНИЕ ПО ТИПАМ ПОЗИЦИЙ")
    print("="*70)
    
    long_df = df[df['type'].astype(str).str.contains('LONG', na=False)]
    short_df = df[df['type'].astype(str).str.contains('SHORT', na=False)]
    
    print(f"\nLONG: {len(long_df)} ({len(long_df)/total_trades*100:.1f}%)")
    print(f"SHORT: {len(short_df)} ({len(short_df)/total_trades*100:.1f}%)")
    
    # Анализ по типам
    for side_name, side_df in [('LONG', long_df), ('SHORT', short_df)]:
        if len(side_df) > 0:
            side_profitable = len(side_df[side_df['pnl_value'] > 0])
            side_win_rate = side_profitable / len(side_df) * 100
            side_avg_pnl = side_df['pnl_value'].mean()
            
            print(f"\n  {side_name}:")
            print(f"    Win Rate: {side_win_rate:.1f}%")
            print(f"    Средний PnL: {side_avg_pnl:.3f}%")
            print(f"    Прибыльных: {side_profitable}/{len(side_df)}")
    
    # ==================== ПРИЧИНЫ ЗАКРЫТИЯ ====================
    print("\n" + "="*70)
    print("🚪 ПРИЧИНЫ ЗАКРЫТИЯ ПОЗИЦИЙ")
    print("="*70)
    
    exit_reasons = df['exit_reason'].value_counts()
    for reason, count in exit_reasons.items():
        pct = count / total_trades * 100
        avg_pnl_for_reason = df[df['exit_reason'] == reason]['pnl_value'].mean()
        print(f"\n{reason}: {count} ({pct:.1f}%)")
        print(f"  Средний PnL: {avg_pnl_for_reason:.3f}%")
    
    # ==================== КАЧЕСТВО СДЕЛОК ====================
    print("\n" + "="*70)
    print("⭐ КАЧЕСТВО СДЕЛОК")
    print("="*70)
    
    quality_counts = df['trade_quality'].value_counts()
    for quality, count in quality_counts.items():
        pct = count / total_trades * 100
        avg_pnl_for_quality = df[df['trade_quality'] == quality]['pnl_value'].mean()
        print(f"\n{quality}: {count} ({pct:.1f}%)")
        print(f"  Средний PnL: {avg_pnl_for_quality:.3f}%")
    
    # ==================== RR СТАТИСТИКА ====================
    print("\n" + "="*70)
    print("📊 RR (RISK-REWARD) СТАТИСТИКА")
    print("="*70)
    
    if 'rr_ratio' in df.columns:
        df['rr_value'] = pd.to_numeric(df['rr_ratio'], errors='coerce')
        print(f"\nСредний RR: {df['rr_value'].mean():.2f}")
        print(f"Медианный RR: {df['rr_value'].median():.2f}")
        print(f"Минимальный RR: {df['rr_value'].min():.2f}")
        print(f"Максимальный RR: {df['rr_value'].max():.2f}")
        
        # RR по типам позиций
        print(f"\nRR по типам позиций:")
        if len(long_df) > 0:
            print(f"  LONG средний RR: {pd.to_numeric(long_df['rr_ratio'], errors='coerce').mean():.2f}")
        if len(short_df) > 0:
            print(f"  SHORT средний RR: {pd.to_numeric(short_df['rr_ratio'], errors='coerce').mean():.2f}")
        
        # RR по качеству
        print(f"\nRR по качеству сделок:")
        for quality in df['trade_quality'].unique():
            quality_rr = pd.to_numeric(df[df['trade_quality'] == quality]['rr_ratio'], errors='coerce').mean()
            print(f"  {quality}: {quality_rr:.2f}")
    
    # ==================== ПРОБЛЕМНЫЕ ПАТТЕРНЫ ====================
    print("\n" + "="*70)
    print("⚠️  ВЫЯВЛЕННЫЕ ПРОБЛЕМЫ")
    print("="*70)
    
    problems = []
    recommendations = []
    
    # 1. Низкий Win Rate
    if win_rate < 50:
        problems.append(f"❌ Низкий Win Rate: {win_rate:.1f}% (цель: ≥50%)")
        recommendations.append("💡 Улучшить MTF фильтры входа - больше сделок закрывается по SL")
        recommendations.append("💡 Проверить синхронизацию таймфреймов - возможно конфликты трендов")
    
    # 2. Много SL_INITIAL
    sl_initial_count = len(df[df['exit_reason'] == 'SL_INITIAL'])
    sl_initial_pct = sl_initial_count / total_trades * 100
    if sl_initial_pct > 25:
        problems.append(f"❌ Слишком много SL_INITIAL: {sl_initial_pct:.1f}% ({sl_initial_count} сделок)")
        recommendations.append("💡 Улучшить MTF фильтры входа или увеличить SL расстояние")
        recommendations.append("💡 Проверить фильтры конфликта трендов между ТФ")
    
    # 3. Много SL_TRAILING
    sl_trailing_count = len(df[df['exit_reason'] == 'SL_TRAILING'])
    sl_trailing_pct = sl_trailing_count / total_trades * 100
    if sl_trailing_pct > 40:
        problems.append(f"❌ Слишком много SL_TRAILING: {sl_trailing_pct:.1f}% ({sl_trailing_count} сделок)")
        recommendations.append("💡 Настроить трейлинг-стоп: возможно, он слишком агрессивный")
        recommendations.append("💡 Увеличить trailing_activation_atr для более поздней активации")
    
    # 4. Мало TP закрытий
    tp_count = len(df[df['exit_reason'].astype(str).str.contains('TP', na=False)])
    tp_pct = tp_count / total_trades * 100
    if tp_pct < 20:
        problems.append(f"❌ Мало закрытий по TP: {tp_pct:.1f}% ({tp_count} сделок)")
        recommendations.append("💡 TP уровни могут быть слишком далеко - рассмотреть снижение")
        recommendations.append("💡 Проверить MTF расчет TP - возможно переоценка волатильности")
    
    # 5. Низкий средний RR
    if 'rr_ratio' in df.columns:
        avg_rr = pd.to_numeric(df['rr_ratio'], errors='coerce').mean()
        if avg_rr < 1.8:
            problems.append(f"❌ Низкий средний RR: {avg_rr:.2f} (цель: ≥1.8)")
            recommendations.append("💡 Увеличить TP уровни или уменьшить SL для лучшего RR")
            recommendations.append("💡 Использовать MTF-взвешенный ATR для более точного расчета")
    
    # 6. Дисбаланс LONG/SHORT
    long_pct = len(long_df) / total_trades * 100
    short_pct = len(short_df) / total_trades * 100
    if abs(long_pct - short_pct) > 40:
        problems.append(f"❌ Дисбаланс позиций: LONG {long_pct:.1f}% vs SHORT {short_pct:.1f}%")
        recommendations.append("💡 Увеличить ent_coef или добавить бонусы за разнообразие")
        recommendations.append("💡 Проверить MTF фильтры - возможно они предвзяты к LONG")
    
    # 7. Много VERY_BAD сделок
    very_bad_count = len(df[df['trade_quality'] == 'VERY_BAD'])
    very_bad_pct = very_bad_count / total_trades * 100
    if very_bad_pct > 20:
        problems.append(f"❌ Много VERY_BAD сделок: {very_bad_pct:.1f}% ({very_bad_count} сделок)")
        recommendations.append("💡 Ужесточить MTF фильтры входа - слишком много плохих входов")
        recommendations.append("💡 Усилить проверку конфликта трендов между ТФ")
    
    # 8. Средний PnL отрицательный
    if avg_pnl < 0:
        problems.append(f"❌ Отрицательный средний PnL: {avg_pnl:.3f}%")
        recommendations.append("💡 Критично: модель теряет деньги - нужны серьезные изменения")
        recommendations.append("💡 Пересмотреть MTF стратегию - возможно неправильная интерпретация сигналов")
    
    # Вывод проблем
    if problems:
        print("\nНайдено проблем:")
        for i, problem in enumerate(problems, 1):
            print(f"{i}. {problem}")
    else:
        print("\n✅ Серьезных проблем не обнаружено!")
    
    # ==================== РЕКОМЕНДАЦИИ ====================
    print("\n" + "="*70)
    print("💡 РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ")
    print("="*70)
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    else:
        print("\n✅ Все показатели в норме!")
    
    # ==================== КОНКРЕТНЫЕ ПАРАМЕТРЫ ====================
    print("\n" + "="*70)
    print("⚙️  РЕКОМЕНДАЦИИ ПО ПАРАМЕТРАМ MTF")
    print("="*70)
    
    # Анализ для рекомендаций параметров
    param_recs = []
    
    # Если много SL_INITIAL - увеличить min_sl_percent или улучшить фильтры
    if sl_initial_pct > 25:
        param_recs.append("📌 Увеличить MTF_MIN_ABSOLUTE_ATR с 120.0 до 150.0")
        param_recs.append("📌 Ужесточить MTF_MIN_ADX с 27.0 до 30.0")
        param_recs.append("📌 Усилить проверку конфликта трендов")
    
    # Если много SL_TRAILING - настроить трейлинг
    if sl_trailing_pct > 40:
        param_recs.append("📌 Увеличить MTF_TRAILING_ACTIVATION_ATR с 0.40 до 0.45-0.50")
        param_recs.append("📌 Увеличить MTF_TRAILING_DISTANCE_ATR с 0.50 до 0.55-0.60")
    
    # Если мало TP - снизить TP уровни
    if tp_pct < 20:
        param_recs.append("📌 Снизить MTF_TP_LEVELS с [2.5, 3.0, 3.8] до [2.2, 2.8, 3.5]")
    
    # Если низкий RR - увеличить TP или уменьшить SL
    if 'rr_ratio' in df.columns:
        avg_rr = pd.to_numeric(df['rr_ratio'], errors='coerce').mean()
        if avg_rr < 1.8:
            param_recs.append("📌 Увеличить MTF_TP_LEVELS[0] с 2.5 до 2.8-3.0")
            param_recs.append("📌 Или уменьшить atr_multiplier для SL с 2.2 до 2.0")
    
    # Если дисбаланс LONG/SHORT
    if abs(long_pct - short_pct) > 40:
        param_recs.append("📌 Увеличить ent_coef с 0.05 до 0.07-0.10")
        param_recs.append("📌 Усилить бонусы за SHORT позиции в MTF reward функции")
        param_recs.append("📌 Проверить MTF_SHORT_RSI_MIN/MAX - возможно слишком строгие")
    
    # Если низкий Win Rate
    if win_rate < 50:
        param_recs.append("📌 Ужесточить MTF_MIN_ADX до 30.0")
        param_recs.append("📌 Увеличить MTF_MIN_VOLUME_SPIKE с 1.6 до 1.8")
        param_recs.append("📌 Уменьшить MTF_MAX_VOLATILITY_RATIO с 1.2 до 1.1")
        param_recs.append("📌 Усилить фильтры конфликта трендов между ТФ")
    
    if param_recs:
        print("\nРекомендуемые изменения параметров:")
        for rec in param_recs:
            print(f"  {rec}")
    else:
        print("\n✅ Параметры MTF в целом оптимальны")
    
    # ==================== ТОП ПРОБЛЕМНЫХ СДЕЛОК ====================
    print("\n" + "="*70)
    print("🔍 ТОП-10 ХУДШИХ СДЕЛОК")
    print("="*70)
    
    worst_trades = df.nsmallest(10, 'pnl_value')[['type', 'pnl_percent', 'exit_reason', 'trade_quality', 'rr_ratio']]
    print("\n" + worst_trades.to_string(index=False))
    
    # ==================== ТОП ЛУЧШИХ СДЕЛОК ====================
    print("\n" + "="*70)
    print("🌟 ТОП-10 ЛУЧШИХ СДЕЛОК")
    print("="*70)
    
    best_trades = df.nlargest(10, 'pnl_value')[['type', 'pnl_percent', 'exit_reason', 'trade_quality', 'rr_ratio']]
    print("\n" + best_trades.to_string(index=False))
    
    # ==================== ИТОГОВАЯ ОЦЕНКА ====================
    print("\n" + "="*70)
    print("📋 ИТОГОВАЯ ОЦЕНКА")
    print("="*70)
    
    score = 0
    max_score = 8
    
    if win_rate >= 50:
        score += 1
    if sl_initial_pct <= 25:
        score += 1
    if sl_trailing_pct <= 40:
        score += 1
    if tp_pct >= 20:
        score += 1
    if 'rr_ratio' in df.columns:
        avg_rr = pd.to_numeric(df['rr_ratio'], errors='coerce').mean()
        if avg_rr >= 1.8:
            score += 1
    if abs(long_pct - short_pct) <= 40:
        score += 1
    if very_bad_pct <= 20:
        score += 1
    if avg_pnl >= 0:
        score += 1
    
    score_pct = (score / max_score) * 100
    
    print(f"\nОценка: {score}/{max_score} ({score_pct:.0f}%)")
    
    if score_pct >= 75:
        print("✅ Отличные результаты! MTF модель работает хорошо.")
    elif score_pct >= 50:
        print("⚠️  Средние результаты. Есть что улучшить в MTF стратегии.")
    else:
        print("❌ Низкие результаты. Требуются серьезные изменения в MTF подходе.")
    
    # ==================== СРАВНЕНИЕ С V17 ====================
    print("\n" + "="*70)
    print("📊 СРАВНЕНИЕ С БАЗОВОЙ ВЕРСИЕЙ V17")
    print("="*70)
    
    print("\n⚠️  Для полного сравнения запустите анализ V17:")
    print("   python bot/analyze_training_detailed.py")
    print("\nОжидаемые улучшения от MTF:")
    print("  ✅ Win Rate: +5-10% (за счет фильтрации конфликтов трендов)")
    print("  ✅ Средний RR: +0.2-0.3 (за счет MTF-оптимизированных TP/SL)")
    print("  ✅ Меньше VERY_BAD сделок (за счет строгих MTF фильтров)")
    print("  ⚠️  Меньше сделок в целом (но выше качество)")
    
    print("\n" + "="*70)
    print("✅ Анализ завершен!")
    print("="*70)


if __name__ == "__main__":
    analyze_training_logs()
