"""
Детальный анализ логов обучения V17 Optimized
Выявляет проблемы и дает рекомендации по улучшению
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def analyze_training_logs(log_file: str = '../logs/v17_optimized_v2/train_v17_log.csv'):
    """Детальный анализ логов обучения"""
    
    if not os.path.exists(log_file):
        print(f"❌ Файл {log_file} не найден!")
        print(f"   Проверьте путь: {os.path.abspath(log_file)}")
        return
    
    print("="*70)
    print("📊 ДЕТАЛЬНЫЙ АНАЛИЗ ЛОГОВ ОБУЧЕНИЯ V17 OPTIMIZED")
    print("="*70)
    
    # Загрузка данных
    try:
        df = pd.read_csv(log_file)
        print(f"\n✅ Загружено {len(df)} сделок из {log_file}")
    except Exception as e:
        print(f"❌ Ошибка загрузки файла: {e}")
        return
    
    if len(df) == 0:
        print("❌ Файл пуст!")
        return
    
    # Проверяем и обрабатываем колонку PnL
    pnl_col = 'pnl_percent' if 'pnl_percent' in df.columns else 'pnl_pct'
    
    # Конвертируем pnl_percent в числовой формат, если это строка с "%"
    if pnl_col in df.columns:
        if df[pnl_col].dtype == 'object':
            # Удаляем "%" и пробелы, конвертируем в float
            df[pnl_col] = df[pnl_col].astype(str).str.replace('%', '').str.replace(' ', '').str.strip()
            df[pnl_col] = pd.to_numeric(df[pnl_col], errors='coerce')
        else:
            df[pnl_col] = pd.to_numeric(df[pnl_col], errors='coerce')
    
    # Удаляем строки с NaN в pnl_col
    df = df[df[pnl_col].notna()].copy()
    
    if len(df) == 0:
        print("❌ Нет валидных данных после обработки!")
        return
    
    # ==================== БАЗОВАЯ СТАТИСТИКА ====================
    print("\n" + "="*70)
    print("📈 БАЗОВАЯ СТАТИСТИКА")
    print("="*70)
    
    total_trades = len(df)
    profitable = len(df[df[pnl_col] > 0])
    losses = len(df[df[pnl_col] < 0])
    breakeven = len(df[df[pnl_col] == 0])
    
    win_rate = profitable / total_trades * 100 if total_trades > 0 else 0
    avg_pnl = df[pnl_col].mean()
    total_pnl = df[pnl_col].sum()
    
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
    
    # Проверяем название колонки для типа позиции
    side_col = 'type' if 'type' in df.columns else 'side'
    
    long_df = df[df[side_col] == 'LONG']
    short_df = df[df[side_col] == 'SHORT']
    
    print(f"\nLONG: {len(long_df)} ({len(long_df)/total_trades*100:.1f}%)")
    print(f"SHORT: {len(short_df)} ({len(short_df)/total_trades*100:.1f}%)")
    
    # Анализ по типам
    for side_name, side_df in [('LONG', long_df), ('SHORT', short_df)]:
        if len(side_df) > 0:
            side_profitable = len(side_df[side_df[pnl_col] > 0])
            side_win_rate = side_profitable / len(side_df) * 100
            side_avg_pnl = side_df[pnl_col].mean()
            
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
        avg_pnl_for_reason = df[df['exit_reason'] == reason][pnl_col].mean()
        print(f"\n{reason}: {count} ({pct:.1f}%)")
        print(f"  Средний PnL: {avg_pnl_for_reason:.3f}%")
    
    # ==================== КАЧЕСТВО СДЕЛОК ====================
    print("\n" + "="*70)
    print("⭐ КАЧЕСТВО СДЕЛОК")
    print("="*70)
    
    # Проверяем название колонки качества
    quality_col = 'trade_quality' if 'trade_quality' in df.columns else 'quality'
    
    quality_counts = df[quality_col].value_counts()
    for quality, count in quality_counts.items():
        pct = count / total_trades * 100
        avg_pnl_for_quality = df[df[quality_col] == quality][pnl_col].mean()
        print(f"\n{quality}: {count} ({pct:.1f}%)")
        print(f"  Средний PnL: {avg_pnl_for_quality:.3f}%")
    
    # ==================== RR СТАТИСТИКА ====================
    print("\n" + "="*70)
    print("📊 RR (RISK-REWARD) СТАТИСТИКА")
    print("="*70)
    
    if 'rr_ratio' in df.columns:
        print(f"\nСредний RR: {df['rr_ratio'].mean():.2f}")
        print(f"Медианный RR: {df['rr_ratio'].median():.2f}")
        print(f"Минимальный RR: {df['rr_ratio'].min():.2f}")
        print(f"Максимальный RR: {df['rr_ratio'].max():.2f}")
        
        # RR по типам позиций
        print(f"\nRR по типам позиций:")
        print(f"  LONG средний RR: {long_df['rr_ratio'].mean():.2f}" if len(long_df) > 0 else "  LONG: нет данных")
        print(f"  SHORT средний RR: {short_df['rr_ratio'].mean():.2f}" if len(short_df) > 0 else "  SHORT: нет данных")
        
        # RR по качеству
        print(f"\nRR по качеству сделок:")
        for quality in df[quality_col].unique():
            quality_rr = df[df[quality_col] == quality]['rr_ratio'].mean()
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
        recommendations.append("💡 Улучшить фильтры входа - больше сделок закрывается по SL")
    
    # 2. Много SL_INITIAL
    sl_initial_count = len(df[df['exit_reason'] == 'SL_INITIAL'])
    sl_initial_pct = sl_initial_count / total_trades * 100
    if sl_initial_pct > 30:
        problems.append(f"❌ Слишком много SL_INITIAL: {sl_initial_pct:.1f}% ({sl_initial_count} сделок)")
        recommendations.append("💡 Улучшить фильтры входа или увеличить SL расстояние")
    
    # 3. Много SL_TRAILING
    sl_trailing_count = len(df[df['exit_reason'] == 'SL_TRAILING'])
    sl_trailing_pct = sl_trailing_count / total_trades * 100
    if sl_trailing_pct > 40:
        problems.append(f"❌ Слишком много SL_TRAILING: {sl_trailing_pct:.1f}% ({sl_trailing_count} сделок)")
        recommendations.append("💡 Настроить трейлинг-стоп: возможно, он слишком агрессивный")
    
    # 4. Мало TP закрытий
    tp_count = len(df[df['exit_reason'].str.contains('TP', na=False)])
    tp_pct = tp_count / total_trades * 100
    if tp_pct < 20:
        problems.append(f"❌ Мало закрытий по TP: {tp_pct:.1f}% ({tp_count} сделок)")
        recommendations.append("💡 TP уровни могут быть слишком далеко - рассмотреть снижение")
    
    # 5. Низкий средний RR
    if 'rr_ratio' in df.columns:
        avg_rr = df['rr_ratio'].mean()
        if avg_rr < 1.8:
            problems.append(f"❌ Низкий средний RR: {avg_rr:.2f} (цель: ≥1.8)")
            recommendations.append("💡 Увеличить TP уровни или уменьшить SL для лучшего RR")
    
    # 6. Дисбаланс LONG/SHORT
    long_pct = len(long_df) / total_trades * 100
    short_pct = len(short_df) / total_trades * 100
    if abs(long_pct - short_pct) > 40:
        problems.append(f"❌ Дисбаланс позиций: LONG {long_pct:.1f}% vs SHORT {short_pct:.1f}%")
        recommendations.append("💡 Увеличить ent_coef или добавить бонусы за разнообразие")
    
    # 7. Много VERY_BAD сделок
    very_bad_count = len(df[df[quality_col] == 'VERY_BAD'])
    very_bad_pct = very_bad_count / total_trades * 100
    if very_bad_pct > 20:
        problems.append(f"❌ Много VERY_BAD сделок: {very_bad_pct:.1f}% ({very_bad_count} сделок)")
        recommendations.append("💡 Ужесточить фильтры входа - слишком много плохих входов")
    
    # 8. Средний PnL отрицательный
    if avg_pnl < 0:
        problems.append(f"❌ Отрицательный средний PnL: {avg_pnl:.3f}%")
        recommendations.append("💡 Критично: модель теряет деньги - нужны серьезные изменения")
    
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
    print("⚙️  РЕКОМЕНДАЦИИ ПО ПАРАМЕТРАМ")
    print("="*70)
    
    # Анализ для рекомендаций параметров
    param_recs = []
    
    # Если много SL_INITIAL - увеличить min_sl_percent или улучшить фильтры
    if sl_initial_pct > 30:
        param_recs.append("📌 Увеличить min_sl_percent с 0.003 до 0.004-0.005")
        param_recs.append("📌 Ужесточить min_trend_strength с 0.45 до 0.50")
    
    # Если много SL_TRAILING - настроить трейлинг
    if sl_trailing_pct > 40:
        param_recs.append("📌 Увеличить trailing_activation_atr с 0.20 до 0.25-0.30")
        param_recs.append("📌 Увеличить trailing_distance_atr с 0.30 до 0.35-0.40")
    
    # Если мало TP - снизить TP уровни
    if tp_pct < 20:
        param_recs.append("📌 Снизить tp_levels с [2.0, 3.0, 4.0] до [1.8, 2.5, 3.5]")
    
    # Если низкий RR - увеличить TP или уменьшить SL
    if 'rr_ratio' in df.columns and df['rr_ratio'].mean() < 1.8:
        param_recs.append("📌 Увеличить tp_levels[0] с 2.0 до 2.2-2.5")
        param_recs.append("📌 Или уменьшить atr_multiplier для SL с 2.5 до 2.2")
    
    # Если дисбаланс LONG/SHORT
    if abs(long_pct - short_pct) > 40:
        param_recs.append("📌 Увеличить ent_coef с 0.05 до 0.07-0.10")
        param_recs.append("📌 Усилить бонусы за SHORT позиции в reward функции")
    
    # Если низкий Win Rate
    if win_rate < 50:
        param_recs.append("📌 Ужесточить min_trend_strength до 0.50-0.55")
        param_recs.append("📌 Увеличить min_volume_ratio с 1.2 до 1.3-1.4")
        param_recs.append("📌 Уменьшить max_volatility_ratio с 1.8 до 1.6")
    
    if param_recs:
        print("\nРекомендуемые изменения параметров:")
        for rec in param_recs:
            print(f"  {rec}")
    else:
        print("\n✅ Параметры в целом оптимальны")
    
    # ==================== ТОП ПРОБЛЕМНЫХ СДЕЛОК ====================
    print("\n" + "="*70)
    print("🔍 ТОП-10 ХУДШИХ СДЕЛОК")
    print("="*70)
    
    worst_trades = df.nsmallest(10, pnl_col)[[side_col, pnl_col, 'exit_reason', quality_col, 'rr_ratio']]
    print("\n" + worst_trades.to_string(index=False))
    
    # ==================== ТОП ЛУЧШИХ СДЕЛОК ====================
    print("\n" + "="*70)
    print("🌟 ТОП-10 ЛУЧШИХ СДЕЛОК")
    print("="*70)
    
    best_trades = df.nlargest(10, pnl_col)[[side_col, pnl_col, 'exit_reason', quality_col, 'rr_ratio']]
    print("\n" + best_trades.to_string(index=False))
    
    # ==================== ИТОГОВАЯ ОЦЕНКА ====================
    print("\n" + "="*70)
    print("📋 ИТОГОВАЯ ОЦЕНКА")
    print("="*70)
    
    score = 0
    max_score = 8
    
    if win_rate >= 50:
        score += 1
    if sl_initial_pct <= 30:
        score += 1
    if sl_trailing_pct <= 40:
        score += 1
    if tp_pct >= 20:
        score += 1
    if 'rr_ratio' in df.columns and df['rr_ratio'].mean() >= 1.8:
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
        print("✅ Отличные результаты! Модель работает хорошо.")
    elif score_pct >= 50:
        print("⚠️  Средние результаты. Есть что улучшить.")
    else:
        print("❌ Низкие результаты. Требуются серьезные изменения.")
    
    # ==================== АНАЛИЗ ПРИЗНАКОВ ====================
    print("\n" + "="*70)
    print("🔬 АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ")
    print("="*70)
    
    # Определяем правильные названия колонок
    pnl_col_name = pnl_col
    side_col_name = 'type' if 'type' in df.columns else ('side' if 'side' in df.columns else None)
    quality_col_name = 'trade_quality' if 'trade_quality' in df.columns else ('quality' if 'quality' in df.columns else None)
    
    if side_col_name and quality_col_name:
        analyze_feature_importance(df, log_file, pnl_col_name, side_col_name, quality_col_name)
    else:
        print("\n⚠️  Не найдены необходимые колонки для анализа признаков")
        print(f"   Найдены колонки: {df.columns.tolist()}")
    
    print("\n" + "="*70)
    print("✅ Анализ завершен!")
    print("="*70)


def analyze_feature_importance(trades_df: pd.DataFrame, log_file: str, pnl_col: str, side_col: str, quality_col: str):
    """Анализ важности признаков для понимания, какие работают лучше"""
    
    # Загружаем исходные данные
    data_file = './data/btc_15m.csv'
    if not os.path.exists(data_file):
        data_file = '../data/btc_15m.csv'
    
    if not os.path.exists(data_file):
        print("\n⚠️  Исходные данные не найдены, пропускаю анализ признаков")
        print(f"   Ожидаемый путь: {os.path.abspath(data_file)}")
        return
    
    try:
        print(f"\n📥 Загрузка исходных данных из {data_file}...")
        market_data = pd.read_csv(data_file)
        print(f"✅ Загружено {len(market_data)} строк")
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        return
    
    # Нормализуем названия колонок
    column_mapping = {
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume',
        'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'
    }
    market_data.rename(columns={k: v for k, v in column_mapping.items() if k in market_data.columns}, inplace=True)
    
    # Создаем базовые признаки если их нет
    if 'atr' not in market_data.columns:
        high_low = market_data['high'] - market_data['low']
        market_data['atr'] = high_low.rolling(window=14, min_periods=1).mean()
    
    if 'rsi_norm' not in market_data.columns:
        delta = market_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        market_data['rsi_norm'] = (rsi - 50) / 50
        market_data['rsi'] = rsi
    
    if 'trend_bias_1h' not in market_data.columns:
        market_data['trend_bias_1h'] = np.sin(np.arange(len(market_data)) * 0.01) * 0.8
    
    if 'volatility_ratio' not in market_data.columns:
        returns = market_data['close'].pct_change()
        market_data['volatility_ratio'] = returns.rolling(20).std().fillna(1.5)
    
    if 'volume_ratio' not in market_data.columns:
        market_data['volume_ratio'] = market_data['volume'] / market_data['volume'].rolling(20).mean().fillna(1.2)
    
    # Список признаков для анализа
    feature_cols = [
        'rsi_norm', 'rsi', 'trend_bias_1h', 'volatility_ratio', 'volume_ratio',
        'atr', 'close', 'volume'
    ]
    
    # Добавляем дополнительные признаки если они есть
    additional_features = [
        'bb_position', 'momentum', 'adx', 'log_ret', 'returns',
        'high_low_ratio', 'close_open_ratio',
        'tp_up_atr_1', 'tp_up_prob_1', 'tp_down_atr_1', 'tp_down_prob_1',
        'sl_up_atr', 'sl_down_atr'
    ]
    
    for feat in additional_features:
        if feat in market_data.columns:
            feature_cols.append(feat)
    
    # Фильтруем только существующие колонки
    feature_cols = [col for col in feature_cols if col in market_data.columns]
    
    print(f"\n📊 Анализирую {len(feature_cols)} признаков...")
    
    # Сопоставляем сделки с данными рынка по шагам
    trades_with_features = []
    
    for idx, trade in trades_df.iterrows():
        step = trade.get('step', None)
        if step is None or pd.isna(step):
            continue
        
        step = int(step)
        if step < len(market_data):
            pnl_value = trade.get(pnl_col, 0)
            # Обрабатываем строковые значения с "%"
            if isinstance(pnl_value, str):
                pnl_value = float(str(pnl_value).replace('%', '').replace(' ', '').strip() or 0)
            else:
                pnl_value = float(pnl_value) if pd.notna(pnl_value) else 0.0
            
            trade_features = {
                'pnl': pnl_value,
                'profitable': 1 if pnl_value > 0 else 0,
                'step': step
            }
            
            # Добавляем значения признаков на момент входа
            for feat in feature_cols:
                if feat in market_data.columns:
                    value = market_data.iloc[step][feat]
                    if pd.notna(value):
                        trade_features[feat] = float(value)
                    else:
                        trade_features[feat] = 0.0
            
            trades_with_features.append(trade_features)
    
    if len(trades_with_features) == 0:
        print("⚠️  Не удалось сопоставить сделки с данными рынка")
        return
    
    features_df = pd.DataFrame(trades_with_features)
    
    # Анализ корреляции признаков с PnL
    print("\n" + "-"*70)
    print("📈 КОРРЕЛЯЦИЯ ПРИЗНАКОВ С PnL")
    print("-"*70)
    
    correlations = {}
    for feat in feature_cols:
        if feat in features_df.columns:
            corr = features_df[feat].corr(features_df['pnl'])
            if pd.notna(corr):
                correlations[feat] = corr
    
    # Сортируем по абсолютной корреляции
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print("\nТоп-10 признаков по корреляции с PnL:")
    for i, (feat, corr) in enumerate(sorted_corr[:10], 1):
        direction = "📈" if corr > 0 else "📉"
        print(f"  {i:2d}. {feat:25s}: {corr:7.4f} {direction}")
    
    # Сравнение средних значений для прибыльных и убыточных сделок
    print("\n" + "-"*70)
    print("⚖️  СРАВНЕНИЕ ПРИЗНАКОВ: ПРИБЫЛЬНЫЕ vs УБЫТОЧНЫЕ")
    print("-"*70)
    
    profitable_df = features_df[features_df['profitable'] == 1]
    unprofitable_df = features_df[features_df['profitable'] == 0]
    
    if len(profitable_df) > 0 and len(unprofitable_df) > 0:
        feature_differences = []
        
        for feat in feature_cols:
            if feat in features_df.columns:
                prof_mean = profitable_df[feat].mean()
                unprof_mean = unprofitable_df[feat].mean()
                
                if pd.notna(prof_mean) and pd.notna(unprof_mean):
                    diff = prof_mean - unprof_mean
                    diff_pct = (diff / abs(unprof_mean) * 100) if unprof_mean != 0 else 0
                    feature_differences.append((feat, prof_mean, unprof_mean, diff, diff_pct))
        
        # Сортируем по абсолютной разнице
        feature_differences.sort(key=lambda x: abs(x[3]), reverse=True)
        
        print("\nТоп-10 признаков с наибольшей разницей:")
        print(f"{'Признак':<25} {'Прибыльные':>12} {'Убыточные':>12} {'Разница':>12} {'%':>8}")
        print("-" * 75)
        
        for feat, prof_mean, unprof_mean, diff, diff_pct in feature_differences[:10]:
            print(f"{feat:<25} {prof_mean:>12.4f} {unprof_mean:>12.4f} {diff:>12.4f} {diff_pct:>7.1f}%")
    
    # Анализ важности признаков для Win Rate
    print("\n" + "-"*70)
    print("🎯 ВАЖНОСТЬ ПРИЗНАКОВ ДЛЯ WIN RATE")
    print("-"*70)
    
    # Разбиваем каждый признак на квартили и смотрим Win Rate в каждом
    feature_importance = []
    
    for feat in feature_cols[:15]:  # Анализируем топ-15 признаков
        if feat not in features_df.columns:
            continue
        
        # Разбиваем на квартили
        q1 = features_df[feat].quantile(0.25)
        q2 = features_df[feat].quantile(0.50)
        q3 = features_df[feat].quantile(0.75)
        
        # Win Rate в каждом квартиле
        q1_trades = features_df[features_df[feat] <= q1]
        q2_trades = features_df[(features_df[feat] > q1) & (features_df[feat] <= q2)]
        q3_trades = features_df[(features_df[feat] > q2) & (features_df[feat] <= q3)]
        q4_trades = features_df[features_df[feat] > q3]
        
        q1_wr = q1_trades['profitable'].mean() * 100 if len(q1_trades) > 0 else 0
        q2_wr = q2_trades['profitable'].mean() * 100 if len(q2_trades) > 0 else 0
        q3_wr = q3_trades['profitable'].mean() * 100 if len(q3_trades) > 0 else 0
        q4_wr = q4_trades['profitable'].mean() * 100 if len(q4_trades) > 0 else 0
        
        # Разница между лучшим и худшим квартилем
        wr_range = max(q1_wr, q2_wr, q3_wr, q4_wr) - min(q1_wr, q2_wr, q3_wr, q4_wr)
        
        feature_importance.append((feat, wr_range, q1_wr, q2_wr, q3_wr, q4_wr))
    
    # Сортируем по разнице Win Rate
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("\nТоп-10 признаков по влиянию на Win Rate:")
    print(f"{'Признак':<25} {'Разница WR':>12} {'Q1':>8} {'Q2':>8} {'Q3':>8} {'Q4':>8}")
    print("-" * 75)
    
    for feat, wr_range, q1_wr, q2_wr, q3_wr, q4_wr in feature_importance[:10]:
        print(f"{feat:<25} {wr_range:>11.1f}% {q1_wr:>7.1f}% {q2_wr:>7.1f}% {q3_wr:>7.1f}% {q4_wr:>7.1f}%")
    
    # Рекомендации по признакам
    print("\n" + "-"*70)
    print("💡 РЕКОМЕНДАЦИИ ПО ПРИЗНАКАМ")
    print("-"*70)
    
    recommendations = []
    
    # Если признак имеет высокую корреляцию с PnL
    if len(sorted_corr) > 0:
        top_feat = sorted_corr[0][0]
        top_corr = sorted_corr[0][1]
        if abs(top_corr) > 0.1:
            recommendations.append(f"✅ Признак '{top_feat}' имеет сильную корреляцию с PnL ({top_corr:.3f})")
            recommendations.append(f"   Рекомендуется использовать его в фильтрах входа")
    
    # Если признак сильно различается между прибыльными и убыточными
    if len(feature_differences) > 0:
        top_diff_feat = feature_differences[0][0]
        top_diff = feature_differences[0][3]
        if abs(top_diff) > 0.01:
            recommendations.append(f"✅ Признак '{top_diff_feat}' сильно различается между прибыльными и убыточными")
            recommendations.append(f"   Разница: {top_diff:.4f} - можно использовать как фильтр")
    
    # Если признак сильно влияет на Win Rate
    if len(feature_importance) > 0:
        top_imp_feat = feature_importance[0][0]
        top_imp_range = feature_importance[0][1]
        if top_imp_range > 10:
            recommendations.append(f"✅ Признак '{top_imp_feat}' сильно влияет на Win Rate (разница {top_imp_range:.1f}%)")
            recommendations.append(f"   Рекомендуется добавить его в фильтры входа")
    
    if recommendations:
        for rec in recommendations:
            print(f"  {rec}")
    else:
        print("  ⚠️  Недостаточно данных для рекомендаций")


if __name__ == "__main__":
    # Проверяем разные возможные пути
    possible_paths = [
        '../logs/v17_optimized_v2/train_v17_log.csv',
        './logs/v17_optimized_v2/train_v17_log.csv',
        'logs/v17_optimized_v2/train_v17_log.csv',
    ]
    
    log_file = None
    for path in possible_paths:
        if os.path.exists(path):
            log_file = path
            break
    
    if log_file:
        analyze_training_logs(log_file)
    else:
        print("❌ Файл логов не найден!")
        print("Проверьте пути:")
        for path in possible_paths:
            print(f"  - {os.path.abspath(path)}")
