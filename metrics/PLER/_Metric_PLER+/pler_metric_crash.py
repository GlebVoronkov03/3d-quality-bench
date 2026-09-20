# pler_metric.py - ОБНОВЛЕННАЯ ВЕРСИЯ
import numpy as np
import open3d as o3d
import time
import math
import logging
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from pathlib import Path
import concurrent.futures
import threading
from threading import Lock
import multiprocessing as mp

# Импорт новых модулей
from mesh_loader import UniversalMeshLoader
from cache_manager import ComputationCache
from gpu_accelerator import GPUAccelerator

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PLERConfig:
    """Конфигурация метрики PLER"""
    min_rays: int = 1000
    max_rays: int = 20000
    adaptive_sampling: bool = True
    cache_rays: bool = True
    precision: str = 'float64'
    convergence_threshold: float = 0.1
    topology_analysis: bool = True
    texture_analysis: bool = False
    use_gpu: bool = True
    max_workers: int = 0  # 0 = auto
    enable_caching: bool = True

@dataclass
class PLERResult:
    """Результаты вычисления метрики PLER"""
    pler_db: float
    mse: float
    mean_error: float
    max_error: float
    num_rays: int
    computation_time: float
    topology_score: Optional[float] = None
    texture_score: Optional[float] = None
    convergence_achieved: bool = True
    used_gpu: bool = False
    used_caching: bool = False

class PLERMetric:
    def __init__(self, config: Optional[PLERConfig] = None):
        self.config = config or PLERConfig()
        self.ray_cache = {}
        self.ray_lock = Lock()
        
        # Определение числа рабочих потоков
        if self.config.max_workers <= 0:
            self.num_workers = max(1, mp.cpu_count() - 1)
        else:
            self.num_workers = min(self.config.max_workers, mp.cpu_count())
        
        # Инициализация новых компонентов
        self.mesh_loader = UniversalMeshLoader()
        self.cache_manager = ComputationCache() if self.config.enable_caching else None
        self.gpu_accelerator = GPUAccelerator() if self.config.use_gpu else None
        
        self._setup_precision()
        logger.info(f"✅ Инициализирован PLERMetric (потоков: {self.num_workers}, GPU: {self.gpu_accelerator.is_available() if self.gpu_accelerator else False})")
        
    def _setup_precision(self):
        """Настройка точности вычислений"""
        if self.config.precision == 'float64':
            self.dtype = np.float64
        else:
            self.dtype = np.float32

    def _calculate_optimal_rays(self, mesh_complexity: float) -> int:
        """Адаптивное вычисление оптимального количества лучей"""
        if not self.config.adaptive_sampling:
            return self.config.min_rays
            
        # Логарифмическая зависимость от сложности меша
        optimal_rays = int(self.config.min_rays * math.log1p(mesh_complexity))
        return min(optimal_rays, self.config.max_rays)

    def _generate_uniform_rays(self, num_rays: int) -> np.ndarray:
        """Генерация равномерно распределенных лучей с кэшированием и GPU-ускорением"""
        # Пробуем GPU ускорение
        if self.gpu_accelerator and self.gpu_accelerator.is_available():
            gpu_rays = self.gpu_accelerator.generate_rays_gpu(num_rays)
            if gpu_rays is not None:
                return gpu_rays.astype(self.dtype)
        
        # Используем CPU с кэшированием
        with self.ray_lock:
            if self.config.cache_rays and num_rays in self.ray_cache:
                return self.ray_cache[num_rays]
            
            # Фибоначчиева сфера на CPU
            indices = np.arange(0, num_rays, dtype=float) + 0.5
            phi = np.arccos(1 - 2 * indices / num_rays)
            theta = np.pi * (1 + 5**0.5) * indices
            
            x = np.cos(theta) * np.sin(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(phi)
            
            rays = np.stack([x, y, z], axis=1).astype(self.dtype)
            
            if self.config.cache_rays:
                self.ray_cache[num_rays] = rays
                
            return rays

    def _ray_cast_single_chunk(self, mesh: o3d.geometry.TriangleMesh, ray_directions: np.ndarray) -> np.ndarray:
        """Ray casting для одного чанка лучей"""
        try:
            ray_origins = np.zeros_like(ray_directions)
            rays = o3d.core.Tensor(np.hstack([ray_origins, ray_directions]),
                                  dtype=o3d.core.Dtype.Float32)
            
            scene = o3d.t.geometry.RaycastingScene()
            mesh_id = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            
            ans = scene.cast_rays(rays)
            hit_distances = ans['t_hit'].numpy().astype(self.dtype)
            
            # Обработка промахов
            miss_mask = (hit_distances == float('inf')) | (hit_distances > 10.0)
            if np.any(miss_mask):
                mesh_complexity = len(mesh.vertices) / 1000
                miss_threshold = 1.0 + 0.1 * mesh_complexity
                hit_distances[miss_mask] = miss_threshold
            
            return hit_distances
            
        except Exception as e:
            logger.error(f"Ошибка ray casting чанка: {e}")
            # Возвращаем расстояния по умолчанию для проблемного чанка
            return np.ones(len(ray_directions), dtype=self.dtype)

    def _ray_cast_mesh_parallel(self, mesh: o3d.geometry.TriangleMesh, 
                              ray_directions: np.ndarray) -> np.ndarray:
        """Параллельный ray casting с обработкой ошибок"""
        try:
            # Разделяем лучи на чанки для параллельной обработки
            chunk_size = max(100, len(ray_directions) // self.num_workers)
            ray_chunks = [ray_directions[i:i + chunk_size] 
                         for i in range(0, len(ray_directions), chunk_size)]
            
            if len(ray_chunks) == 1:
                # Для маленького количества лучей используем последовательную обработку
                return self._ray_cast_single_chunk(mesh, ray_directions)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(self._ray_cast_single_chunk, mesh, chunk) 
                          for chunk in ray_chunks]
                
                results = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        chunk_result = future.result(timeout=30)  # Таймаут 30 секунд
                        results.append(chunk_result)
                    except Exception as e:
                        logger.error(f"Ошибка в потоке ray casting: {e}")
                        # Возвращаем значения по умолчанию для проблемного чанка
                        default_chunk = np.ones(len(ray_chunks[0]), dtype=self.dtype)
                        results.append(default_chunk)
                
                return np.concatenate(results)
                
        except Exception as e:
            logger.error(f"Критическая ошибка параллельного ray casting: {e}")
            return np.ones(len(ray_directions), dtype=self.dtype)

    def _compute_mesh_complexity(self, mesh: o3d.geometry.TriangleMesh) -> float:
        """Вычисление сложности меша для адаптивной дискретизации"""
        try:
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            
            # Комбинированная метрика сложности
            vertex_complexity = len(vertices) / 1000
            triangle_complexity = len(triangles) / 2000
            bbox_volume = self._compute_bbox_volume(vertices)
            
            return vertex_complexity + triangle_complexity + bbox_volume
        except:
            return 1.0  # Сложность по умолчанию

    def _compute_mesh_complexity_from_file(self, file_path: str) -> float:
        """Вычисление сложности меша из файла без полной загрузки"""
        try:
            # Быстрая оценка сложности по размеру файла и расширению
            file_size = Path(file_path).stat().st_size / 1024  # KB
            return min(file_size / 100, 10.0)  # Нормализованная сложность
        except:
            return 1.0

    def _compute_bbox_volume(self, vertices: np.ndarray) -> float:
        """Вычисление объема ограничивающего параллелепипеда"""
        if len(vertices) == 0:
            return 0.0
        bbox_min = np.min(vertices, axis=0)
        bbox_max = np.max(vertices, axis=0)
        bbox_size = bbox_max - bbox_min
        return np.prod(bbox_size)

    def _compute_pler_from_distances(self, ref_distances: np.ndarray, dist_distances: np.ndarray, 
                                   num_rays: int, start_time: float) -> PLERResult:
        """Вычисление метрики PLER из расстояний"""
        # Пробуем GPU вычисления
        if self.gpu_accelerator and self.gpu_accelerator.is_available():
            gpu_metrics = self.gpu_accelerator.distance_computation_gpu(ref_distances, dist_distances)
            if gpu_metrics is not None:
                mse, mean_error, max_error = gpu_metrics
                used_gpu = True
            else:
                # Fallback на CPU
                errors = np.abs(ref_distances - dist_distances)
                mse = np.mean((ref_distances - dist_distances) ** 2)
                mean_error = np.mean(errors)
                max_error = np.max(errors)
                used_gpu = False
        else:
            # CPU вычисления
            errors = np.abs(ref_distances - dist_distances)
            mse = np.mean((ref_distances - dist_distances) ** 2)
            mean_error = np.mean(errors)
            max_error = np.max(errors)
            used_gpu = False

        # Вычисление L-значений и PLER
        L_ref = 1.0 - ref_distances
        L_dist = 1.0 - dist_distances
        L_min = np.min(L_ref)
        max_val = 1.0 - L_min

        if mse > 1e-10:  # Защита от деления на ноль
            pler_db = 10 * math.log10((max_val ** 2) / mse)
            convergence_achieved = True
        else:
            pler_db = 100.0  # Очень высокое качество
            convergence_achieved = False

        computation_time = time.time() - start_time

        return PLERResult(
            pler_db=pler_db,
            mse=mse,
            mean_error=mean_error,
            max_error=max_error,
            num_rays=num_rays,
            computation_time=computation_time,
            convergence_achieved=convergence_achieved,
            used_gpu=used_gpu,
            used_caching=False
        )

    def compute_pler(self, reference_obj: str, distorted_obj: str,
                    num_rays: Optional[int] = None) -> PLERResult:
        """Улучшенное вычисление метрики PLER с кэшированием и многопоточностью"""
        
        # Проверка существования файлов
        for path in [reference_obj, distorted_obj]:
            if not Path(path).exists():
                raise FileNotFoundError(f"Файл не найден: {path}")
        
        # Определение количества лучей
        if num_rays is None:
            ref_complexity = self._compute_mesh_complexity_from_file(reference_obj)
            num_rays = self._calculate_optimal_rays(ref_complexity)
        
        # Проверка кэша
        if self.cache_manager and self.config.enable_caching:
            cached_result = self.cache_manager.get_cached_result(
                reference_obj, distorted_obj, num_rays, self.config
            )
            if cached_result:
                cached_result.used_caching = True
                return cached_result
        
        logger.info("🔍 Загрузка и нормализация моделей...")
        start_time = time.time()
        
        try:
            # Используем универсальный загрузчик
            ref_mesh, ref_center, ref_scale = self.mesh_loader.load_mesh(reference_obj)
            dist_mesh, dist_center, dist_scale = self.mesh_loader.load_mesh(distorted_obj)
            
            logger.info(f"📊 Загружено: ref={len(ref_mesh.vertices)} вершин, dist={len(dist_mesh.vertices)} вершин")
            logger.info(f"🎯 Используется {num_rays} лучей")
            
            # Генерация лучей (с GPU ускорением если доступно)
            ray_directions = self._generate_uniform_rays(num_rays)
            
            # Параллельный ray casting
            logger.info("📏 Выполнение параллельного ray casting...")
            ref_distances = self._ray_cast_mesh_parallel(ref_mesh, ray_directions)
            dist_distances = self._ray_cast_mesh_parallel(dist_mesh, ray_directions)
            
            # Вычисление метрики PLER
            result = self._compute_pler_from_distances(ref_distances, dist_distances, num_rays, start_time)
            
            # Сохранение в кэш
            if self.cache_manager and self.config.enable_caching:
                self.cache_manager.save_result(result, reference_obj, distorted_obj, num_rays, self.config)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Критическая ошибка вычисления PLER: {e}")
            # Возвращаем результат с ошибкой
            return PLERResult(
                pler_db=0.0,
                mse=1.0,
                mean_error=1.0,
                max_error=1.0,
                num_rays=num_rays,
                computation_time=time.time() - start_time,
                convergence_achieved=False,
                used_gpu=False,
                used_caching=False
            )

    def _print_results(self, result: PLERResult):
        """Красивый вывод результатов"""
        print("\n" + "="*50)
        print("📊 РЕЗУЛЬТАТЫ PLER METRIC:")
        print("="*50)
        print(f"PLER: {result.pler_db:.2f} dB")
        print(f"MSE: {result.mse:.6f}")
        print(f"Средняя ошибка: {result.mean_error:.3f}")
        print(f"Максимальная ошибка: {result.max_error:.3f}")
        print(f"Количество лучей: {result.num_rays}")
        if result.topology_score is not None:
            print(f"Топологическая оценка: {result.topology_score:.1%}")
        print(f"Время вычисления: {result.computation_time:.2f} сек")
        print(f"Использовано GPU: {'✅' if result.used_gpu else '❌'}")
        print(f"Использован кэш: {'✅' if result.used_caching else '❌'}")
        
        # Интерпретация результатов
        print("\n📈 ИНТЕРПРЕТАЦИЯ:")
        if result.pler_db >= 60:
            print("✅ Отличное качество - изменения практически не заметны")
        elif result.pler_db >= 40:
            print("⚠️ Хорошее качество - незначительные изменения")
        elif result.pler_db >= 20:
            print("🔶 Удовлетворительное качество - заметные изменения")
        else:
            print("❌ Низкое качество - значительные искажения")
        print("="*50)

    def get_system_info(self) -> Dict[str, any]:
        """Получение информации о системе"""
        info = {
            'cpu_cores': mp.cpu_count(),
            'worker_threads': self.num_workers,
            'gpu_available': self.gpu_accelerator.is_available() if self.gpu_accelerator else False,
            'cache_enabled': self.config.enable_caching,
            'supported_formats': list(self.mesh_loader.get_supported_formats())
        }
        
        if self.gpu_accelerator:
            info.update(self.gpu_accelerator.get_gpu_info())
        
        if self.cache_manager:
            info.update(self.cache_manager.get_cache_info())
        
        return info

    def clear_cache(self):
        """Очистка кэшей"""
        self.mesh_loader.clear_cache()
        if self.cache_manager:
            self.cache_manager.clear_cache()
        self.ray_cache.clear()
        logger.info("✅ Все кэши очищены")

# Сохранение обратной совместимости
def main():
    """Основная функция для запуска из командной строки"""
    import sys
    
    if len(sys.argv) < 3:
        print("Использование: python pler_metric.py <reference.obj> <distorted.obj> [num_rays]")
        print("Пример: python pler_metric.py models/reference.obj models/distorted.obj 1000")
        return
    
    reference_path = sys.argv[1]
    distorted_path = sys.argv[2]
    num_rays = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    # Проверка существования файлов
    if not Path(reference_path).exists():
        print(f"❌ Файл не найден: {reference_path}")
        return
    if not Path(distorted_path).exists():
        print(f"❌ Файл не найден: {distorted_path}")
        return
    
    metric = PLERMetric()
    result = metric.compute_pler(reference_path, distorted_path, num_rays)
    metric._print_results(result)

if __name__ == "__main__":
    main()