"""
Скрипт для исправления настроек стратегий в .env
"""

def fix_env_settings():
    """Обновляет настройки для лучшей генерации сигналов"""
    
    # Читаем текущий .env
    with open('.env', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Изменения
    changes = {
        'MOMENTUM_ADX_THRESHOLD': '20.0',  # Было 25.0
        'RANGE_VOLUME_MULT': '2.0',  # Было 1.3
        'RANGE_BB_TOUCH_TOLERANCE_PCT': '0.002',  # 0.2% допуск для касания BB
        'ENABLE_TREND_STRATEGY': 'true',  # Включить TREND стратегию
    }
    
    # Применяем изменения
    new_lines = []
    updated = set()
    
    for line in lines:
        modified = False
        for key, value in changes.items():
            if line.startswith(f'{key}='):
                old_value = line.split('=', 1)[1].strip()
                new_lines.append(f'{key}={value}\n')
                print(f"✅ {key}: {old_value} → {value}")
                updated.add(key)
                modified = True
                break
        if not modified:
            new_lines.append(line)
    
    # Добавляем отсутствующие ключи
    for key, value in changes.items():
        if key not in updated:
            new_lines.append(f'{key}={value}\n')
            print(f"✅ {key}: добавлено = {value}")
    
    # Сохраняем
    with open('.env', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print("\n" + "=" * 80)
    print("✅ НАСТРОЙКИ ОБНОВЛЕНЫ!")
    print("=" * 80)
    print("\n💡 Изменения:")
    print("  1. MOMENTUM_ADX_THRESHOLD: 25.0 → 20.0")
    print("     → Momentum стратегия будет генерировать сигналы при ADX >= 20")
    print("  2. RANGE_VOLUME_MULT: 1.3 → 2.0")
    print("     → FLAT стратегия разрешит больший объем (до 2x)")
    print("  3. RANGE_BB_TOUCH_TOLERANCE_PCT: добавлено 0.002 (0.2%)")
    print("     → FLAT стратегия будет срабатывать при приближении к BB (не требует точного касания)")
    print("  4. ENABLE_TREND_STRATEGY: включена")
    print("     → Дополнительные сигналы от TREND стратегии")
    print("\n🎯 Теперь бот будет генерировать больше сигналов!")
    print("\n⚠️ Перезапустите бота для применения изменений:")
    print("   python main.py")


if __name__ == "__main__":
    fix_env_settings()
