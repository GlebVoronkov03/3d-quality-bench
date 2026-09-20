# test_improvements.py
import time
from pathlib import Path
from pler_metric_crash import PLERMetric, PLERConfig

def test_improvements():
    """Тестирование всех улучшений"""
    print("🧪 Тестирование улучшений PLER...")
    
    # Конфигурация с включенными улучшениями
    config = PLERConfig(
        min_rays=500,
        max_rays=5000,
        use_gpu=True,
        enable_caching=True,
        max_workers=0  # auto
    )
    
    metric = PLERMetric(config)
    
    # Информация о системе
    print("\n📊 ИНФОРМАЦИЯ О СИСТЕМЕ:")
    system_info = metric.get_system_info()
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    # Тестовые модели
    test_models = [
        ("models/simple_reference.obj", "models/simple_distorted.obj"),
        ("models/simple_reference.obj", "models/heavily_distorted.obj")
    ]
    
    for i, (ref, dist) in enumerate(test_models, 1):
        if not Path(ref).exists() or not Path(dist).exists():
            print(f"⚠️ Тестовые модели не найдены, пропускаем тест {i}")
            continue
            
        print(f"\n🔍 ТЕСТ {i}: {Path(ref).name} vs {Path(dist).name}")
        
        # Первый запуск (без кэша)
        start_time = time.time()
        result1 = metric.compute_pler(ref, dist)
        time1 = time.time() - start_time
        
        print(f"  Первый запуск: {result1.pler_db:.2f} dB, {time1:.2f}с, GPU: {result1.used_gpu}, Кэш: {result1.used_caching}")
        
        # Второй запуск (с кэшем)
        start_time = time.time()
        result2 = metric.compute_pler(ref, dist)
        time2 = time.time() - start_time
        
        print(f"  Второй запуск:  {result2.pler_db:.2f} dB, {time2:.2f}с, GPU: {result2.used_gpu}, Кэш: {result2.used_caching}")
        
        # Сравнение производительности
        if time1 > 0:
            speedup = time1 / time2
            print(f"  Ускорение: {speedup:.1f}x")
    
    # Очистка кэша
    metric.clear_cache()
    print("\n✅ Тестирование завершено")

if __name__ == "__main__":
    test_improvements()