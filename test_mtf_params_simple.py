#!/usr/bin/env python3
"""
Упрощенное тестирование обновленных параметров MTF
"""

import sys
sys.path.insert(0, '.')

from bot.mtf_optimized_params import MTF_OPTIMIZED_PARAMS

def test_mtf_params_simple():
    """Проверка параметров MTF"""
    print("=== УПРОЩЕННОЕ ТЕСТИРОВАНИЕ ОБНОВЛЕННЫХ ПАРАМЕТРОВ MTF ===")
    
    # Проверка параметров из mtf_optimized_params.py
    print("\n1. Параметры из MTF_OPTIMIZED_PARAMS:")
    
    expected_params = {
        'min_volatility_ratio': 0.0030,
        'max_volatility_ratio': 1.05,
        'trailing_activation_atr': 0.65,
        'trailing_distance_atr': 0.75,
        'tp_levels': [3.0, 3.5, 4.2],
        'min_adx': 38.0,
        'min_adx_short': 32.0,
        'min_absolute_atr': 200.0,
        'atr_percent_min': 0.0025,
        'min_absolute_volume': 1300.0,
        'min_volume_spike': 2.2,
        'min_volume_spike_short': 1.7,
        'min_1h_adx': 30.0,
        'min_1h_adx_short': 27.0,
        'di_ratio_1h': 1.25,
        'trend_conflict_multiplier': 1.10,
        'require_4h_confirmation': True
    }
    
    all_passed = True
    
    for key, expected in expected_params.items():
        if key in MTF_OPTIMIZED_PARAMS:
            actual = MTF_OPTIMIZED_PARAMS[key]
            if actual == expected:
                print(f"   [OK] {key}: {actual}")
            else:
                print(f"   [FAIL] {key}: {actual} (ожидалось: {expected})")
                all_passed = False
        else:
            print(f"   [FAIL] {key}: отсутствует в MTF_OPTIMIZED_PARAMS")
            all_passed = False
    
    # Проверка обновленных значений (которые мы увеличили)
    print("\n2. Проверка обновленных значений (ужесточение):")
    
    critical_updates = [
        ('min_1h_adx', 30.0, "Увеличено с 28.0 до 30.0"),
        ('min_1h_adx_short', 27.0, "Увеличено с 25.0 до 27.0"),
        ('di_ratio_1h', 1.25, "Увеличено с 1.20 до 1.25"),
    ]
    
    for key, expected_value, description in critical_updates:
        if key in MTF_OPTIMIZED_PARAMS:
            actual = MTF_OPTIMIZED_PARAMS[key]
            if actual == expected_value:
                print(f"   [OK] {key}: {actual} - {description}")
            else:
                print(f"   [FAIL] {key}: {actual} (ожидалось: {expected_value}) - {description}")
                all_passed = False
    
    # Проверка новых параметров
    print("\n3. Проверка новых параметров:")
    new_params = [
        ('trend_conflict_multiplier', 1.10),
        ('require_4h_confirmation', True)
    ]
    
    for key, expected_value in new_params:
        if key in MTF_OPTIMIZED_PARAMS:
            actual = MTF_OPTIMIZED_PARAMS[key]
            if actual == expected_value:
                print(f"   ✓ {key}: {actual} - новый параметр добавлен")
            else:
                print(f"   ✗ {key}: {actual} (ожидалось: {expected_value})")
                all_passed = False
        else:
            print(f"   ✗ {key}: отсутствует новый параметр")
            all_passed = False
    
    print("\n=== РЕЗУЛЬТАТ ТЕСТИРОВАНИЯ ===")
    if all_passed:
        print("✓ Все параметры успешно обновлены согласно рекомендациям анализа")
        return True
    else:
        print("✗ Обнаружены расхождения с ожидаемыми параметрами")
        return False

if __name__ == "__main__":
    success = test_mtf_params_simple()
    sys.exit(0 if success else 1)