# pler_metric.py
import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree
import math
import time
import sys
import logging
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from pathlib import Path

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

class PLERMetric:
    def __init__(self, config: Optional[PLERConfig] = None):
        self.config = config or PLERConfig()
        self.ray_cache = {}
        self._setup_precision()
        
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
        """Генерация равномерно распределенных лучей с кэшированием"""
        if self.config.cache_rays and num_rays in self.ray_cache:
            return self.ray_cache[num_rays]
            
        # Фибоначчиева сфера
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
    
    def _load_and_normalize_mesh(self, obj_path: str) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray, float]:
        """Улучшенная загрузка и нормализация с обработкой ошибок"""
        try:
            # Сначала пробуем загрузить через trimesh (более надежно)
            return self._load_with_trimesh(obj_path)
        except Exception as e:
            logger.warning(f"Trimesh не смог загрузить {obj_path}: {e}")
            # Fallback на упрощенную загрузку
            return self._load_simple_mesh(obj_path)
    
    def _load_simple_mesh(self, obj_path: str) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray, float]:
        """Упрощенная загрузка для сложных OBJ файлов"""
        try:
            logger.info(f"🔄 Используем упрощенную загрузку для {obj_path}")
            
            # Чтение файла вручную
            with open(obj_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            vertices = []
            faces = []
            
            for line in lines:
                if line.startswith('v '):
                    # Вершины
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('f '):
                    # Грани - берем только первые три вершины для треугольников
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        face_vertices = []
                        for i in range(1, min(4, len(parts))):  # Берем только первые 3
                            vertex_part = parts[i].split('/')[0]  # Берем только индекс вершины
                            if vertex_part.isdigit():
                                vertex_index = int(vertex_part) - 1  # OBJ индексы с 1
                                face_vertices.append(vertex_index)
                        if len(face_vertices) == 3:  # Только треугольники
                            faces.append(face_vertices)
            
            if len(vertices) == 0:
                raise ValueError("Нет вершин в файле")
            if len(faces) == 0:
                raise ValueError("Нет граней в файле")
                
            vertices = np.array(vertices, dtype=self.dtype)
            
            # Нормализация
            center = np.mean(vertices, axis=0, dtype=self.dtype)
            vertices_centered = vertices - center
            max_distance = np.max(np.linalg.norm(vertices_centered, axis=1))
            if max_distance <= 1e-10:
                max_distance = 1.0
            vertices_centered /= max_distance
            
            # Создание mesh
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices_centered)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            
            logger.info(f"✅ Упрощенная загрузка: {len(vertices)} вершин, {len(faces)} граней")
            return mesh, center, max_distance
            
        except Exception as e:
            logger.error(f"❌ Упрощенная загрузка не удалась: {e}")
            # Создаем простую сферу как fallback
            return self._create_fallback_mesh()
    
    def _create_fallback_mesh(self) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray, float]:
        """Создание простой сферы как fallback"""
        logger.warning("🔄 Создание fallback сферы")
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=10)
        vertices = np.asarray(mesh.vertices, dtype=self.dtype)
        center = np.zeros(3, dtype=self.dtype)
        return mesh, center, 1.0
    
    def _load_with_trimesh(self, obj_path: str) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray, float]:
        """Резервный метод загрузки через trimesh"""
        try:
            mesh_trimesh = trimesh.load_mesh(obj_path)
            if not hasattr(mesh_trimesh, 'vertices') or len(mesh_trimesh.vertices) == 0:
                raise ValueError("Trimesh загрузил пустой mesh")
                
            vertices = np.asarray(mesh_trimesh.vertices, dtype=self.dtype)
            center = np.mean(vertices, axis=0, dtype=self.dtype)
            vertices_centered = vertices - center
            
            max_distance = np.max(np.linalg.norm(vertices_centered, axis=1))
            if max_distance <= 1e-10:
                max_distance = 1.0
                
            vertices_centered /= max_distance
            
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices_centered)
            
            # Преобразование граней в треугольники
            if hasattr(mesh_trimesh, 'faces'):
                mesh.triangles = o3d.utility.Vector3iVector(mesh_trimesh.faces)
            else:
                # Если нет граней, создаем простую triangulation
                mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=10)
            
            logger.info(f"✅ Trimesh загрузка: {len(vertices)} вершин")
            return mesh, center, max_distance
            
        except Exception as e:
            raise ValueError(f"Trimesh не смог загрузить mesh: {e}")
    
    def _ray_cast_mesh(self, mesh: o3d.geometry.TriangleMesh, 
                      ray_directions: np.ndarray) -> np.ndarray:
        """Улучшенный ray casting с обработкой граничных случаев"""
        try:
            ray_origins = np.zeros_like(ray_directions)
            rays = o3d.core.Tensor(np.hstack([ray_origins, ray_directions]), 
                                  dtype=o3d.core.Dtype.Float32)
            
            scene = o3d.t.geometry.RaycastingScene()
            mesh_id = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            
            ans = scene.cast_rays(rays)
            hit_distances = ans['t_hit'].numpy().astype(self.dtype)
            
            # Улучшенная обработка промахов
            miss_mask = (hit_distances == float('inf')) | (hit_distances > 10.0)
            if np.any(miss_mask):
                # Адаптивный порог на основе сложности модели
                mesh_complexity = len(mesh.vertices) / 1000
                miss_threshold = 1.0 + 0.1 * mesh_complexity
                hit_distances[miss_mask] = miss_threshold
            
            return hit_distances
            
        except Exception as e:
            logger.error(f"❌ Ошибка ray casting: {e}")
            # Возвращаем расстояния по умолчанию
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
    
    def _compute_bbox_volume(self, vertices: np.ndarray) -> float:
        """Вычисление объема ограничивающего параллелепипеда"""
        if len(vertices) == 0:
            return 0.0
        bbox_min = np.min(vertices, axis=0)
        bbox_max = np.max(vertices, axis=0)
        bbox_size = bbox_max - bbox_min
        return np.prod(bbox_size)
    
    def compute_pler(self, reference_obj: str, distorted_obj: str, 
                    num_rays: Optional[int] = None) -> PLERResult:
        """Улучшенное вычисление метрики PLER"""
        logger.info("🔍 Загрузка и нормализация моделей...")
        
        start_time = time.time()
        
        # Загрузка моделей
        ref_mesh, ref_center, ref_scale = self._load_and_normalize_mesh(reference_obj)
        dist_mesh, dist_center, dist_scale = self._load_and_normalize_mesh(distorted_obj)
        
        # Вычисление сложности модели ДО использования в логгере
        ref_complexity = self._compute_mesh_complexity(ref_mesh)
        
        # Адаптивное определение количества лучей
        if num_rays is None:
            num_rays = self._calculate_optimal_rays(ref_complexity)
        
        logger.info(f"📊 Сложность модели: {ref_complexity:.2f}, Лучей: {num_rays}")
        
        # Генерация лучей
        ray_directions = self._generate_uniform_rays(num_rays)
        
        # Ray casting
        logger.info("📏 Выполнение ray casting...")
        ref_distances = self._ray_cast_mesh(ref_mesh, ray_directions)
        dist_distances = self._ray_cast_mesh(dist_mesh, ray_directions)
        
        # Вычисление метрики PLER
        L_ref = 1.0 - ref_distances
        L_dist = 1.0 - dist_distances
        
        L_min = np.min(L_ref)
        max_val = 1.0 - L_min
        
        mse = np.mean((L_ref - L_dist) ** 2)
        
        if mse > 1e-10:  # Защита от деления на ноль
            pler_db = 10 * math.log10((max_val ** 2) / mse)
            convergence_achieved = True
        else:
            pler_db = 100.0  # Очень высокое качество
            convergence_achieved = False
        
        computation_time = time.time() - start_time
        
        # Дополнительный анализ топологии
        topology_score = None
        if self.config.topology_analysis:
            try:
                topology_score = self._compute_topology_score(reference_obj, distorted_obj)
            except Exception as e:
                logger.warning(f"Топологический анализ не удался: {e}")
        
        result = PLERResult(
            pler_db=pler_db,
            mse=mse,
            mean_error=np.mean(np.abs(L_ref - L_dist)),
            max_error=np.max(np.abs(L_ref - L_dist)),
            num_rays=num_rays,
            computation_time=computation_time,
            topology_score=topology_score,
            convergence_achieved=convergence_achieved
        )
        
        # Красивый вывод результатов
        self._print_results(result)
        
        return result
    
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
    
    def _compute_topology_score(self, ref_path: str, dist_path: str) -> float:
        """Вычисление топологического сходства"""
        try:
            # Импорт здесь чтобы избежать циклических зависимостей
            from topology_analyzer import TopologyAnalyzer
            
            analyzer = TopologyAnalyzer()
            return analyzer.compare_topology(ref_path, dist_path)
        except ImportError:
            logger.warning("Модуль topology_analyzer не найден")
            return 0.0
        except Exception as e:
            logger.warning(f"Топологический анализ не удался: {e}")
            return 0.0
    
    def validate_convergence(self, reference_obj: str, distorted_obj: str, 
                           thresholds: List[int] = [1000, 2000, 5000]) -> bool:
        """Проверка сходимости метрики при разном количестве лучей"""
        results = []
        for rays in thresholds:
            try:
                result = self.compute_pler(reference_obj, distorted_obj, rays)
                results.append(result.pler_db)
                print(f"✅ {rays} лучей: PLER = {result.pler_db:.2f} dB")
            except Exception as e:
                print(f"❌ Ошибка для {rays} лучей: {e}")
                results.append(0.0)
        
        # Проверяем, что изменения меньше порога
        if len(results) < 2:
            return False
            
        differences = np.abs(np.diff(results))
        return np.all(differences < self.config.convergence_threshold)

# Сохранение обратной совместимости
def main():
    """Основная функция для запуска из командной строки"""
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

if __name__ == "__main__":
    main()