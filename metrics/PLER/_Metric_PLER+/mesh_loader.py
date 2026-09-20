# mesh_loader.py
import numpy as np
import open3d as o3d
import trimesh
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging
import os

logger = logging.getLogger(__name__)

class UniversalMeshLoader:
    """Универсальный загрузчик 3D моделей с улучшенной обработкой ошибок"""
    
    SUPPORTED_FORMATS = {'.obj', '.stl', '.gltf', '.glb', '.fbx', '.ply', '.dae', '.3ds', '.off'}
    
    def __init__(self):
        self.load_cache = {}
        self._setup_format_handlers()
    
    def _setup_format_handlers(self):
        """Настройка обработчиков для разных форматов"""
        self.format_handlers = {
            '.obj': self._load_with_trimesh,
            '.stl': self._load_with_trimesh,
            '.gltf': self._load_gltf,
            '.glb': self._load_gltf,
            '.fbx': self._load_fbx,
            '.ply': self._load_with_trimesh,
            '.dae': self._load_with_trimesh,
            '.3ds': self._load_with_trimesh,
            '.off': self._load_with_trimesh
        }
    
    def load_mesh(self, file_path: str) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray, float]:
        """Универсальная загрузка меша с нормализацией и обработкой ошибок"""
        file_path = str(file_path)
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Неподдерживаемый формат: {file_ext}. Поддерживаются: {self.SUPPORTED_FORMATS}")
        
        # Проверка существования файла
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        
        # Проверка размера файла
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise ValueError(f"Файл пуст: {file_path}")
        if file_size > 100 * 1024 * 1024:  # 100 MB limit
            logger.warning(f"Большой файл: {file_size / 1024 / 1024:.1f} MB")
        
        try:
            # Используем кэш для избежания повторной загрузки
            cache_key = f"{file_path}_{file_size}"
            if cache_key in self.load_cache:
                return self.load_cache[cache_key]
            
            # Выбор обработчика по формату
            handler = self.format_handlers.get(file_ext, self._load_with_trimesh)
            mesh_o3d, center, scale = handler(file_path)
            
            # Кэшируем результат
            self.load_cache[cache_key] = (mesh_o3d, center, scale)
            
            logger.info(f"✅ Успешно загружено: {Path(file_path).name} - {len(mesh_o3d.vertices)} вершин, {len(mesh_o3d.triangles)} граней")
            return mesh_o3d, center, scale
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки {file_path}: {e}")
            # Fallback на создание простой сферы
            return self._create_fallback_mesh()
    
    def _load_with_trimesh(self, file_path: str) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray, float]:
        """Загрузка через trimesh (универсальный метод)"""
        try:
            mesh_trimesh = trimesh.load_mesh(file_path, force='mesh')
            
            # Валидация меша
            self._validate_mesh(mesh_trimesh, file_path)
            
            # Конвертация в треугольники если нужно
            if not self._is_triangular(mesh_trimesh):
                logger.info(f"🔺 Триангуляция меша: {file_path}")
                mesh_trimesh = mesh_trimesh.triangulate()
            
            vertices = np.asarray(mesh_trimesh.vertices, dtype=np.float64)
            faces = np.asarray(mesh_trimesh.faces, dtype=np.int32)
            
            return self._create_o3d_mesh(vertices, faces)
            
        except Exception as e:
            logger.error(f"Ошибка trimesh загрузки {file_path}: {e}")
            raise
    
    def _load_gltf(self, file_path: str) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray, float]:
        """Специализированная загрузка GLTF/GLB"""
        try:
            # Open3D хорошо работает с GLTF
            mesh_o3d = o3d.io.read_triangle_mesh(file_path)
            if len(mesh_o3d.vertices) == 0:
                raise ValueError("Open3D не смог загрузить GLTF")
            
            vertices = np.asarray(mesh_o3d.vertices, dtype=np.float64)
            center = np.mean(vertices, axis=0)
            vertices_centered = vertices - center
            max_distance = np.max(np.linalg.norm(vertices_centered, axis=1))
            
            if max_distance <= 1e-10:
                max_distance = 1.0
                logger.warning("Обнаружена вырожденная модель")
            
            vertices_centered /= max_distance
            mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices_centered)
            
            return mesh_o3d, center, max_distance
            
        except Exception as e:
            logger.warning(f"Open3D GLTF загрузка не удалась, пробуем trimesh: {e}")
            return self._load_with_trimesh(file_path)
    
    def _load_fbx(self, file_path: str) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray, float]:
        """Загрузка FBX файлов"""
        try:
            # Пробуем загрузить через trimesh
            return self._load_with_trimesh(file_path)
        except Exception as e:
            logger.error(f"Ошибка загрузки FBX {file_path}: {e}")
            raise ValueError(f"Не удалось загрузить FBX файл: {e}")
    
    def _validate_mesh(self, mesh, file_path: str):
        """Валидация загруженного меша"""
        if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
            raise ValueError(f"Нет вершин в файле: {file_path}")
        
        if not hasattr(mesh, 'faces') or len(mesh.faces) == 0:
            raise ValueError(f"Нет граней в файле: {file_path}")
        
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        
        # Проверка на NaN и Inf
        if np.any(np.isnan(vertices)) or np.any(np.isinf(vertices)):
            raise ValueError("Обнаружены NaN или Inf значения в вершинах")
        
        # Проверка индексов граней
        if np.any(faces < 0) or np.any(faces >= len(vertices)):
            raise ValueError("Некорректные индексы в гранях")
        
        logger.debug(f"Валидация пройдена: {len(vertices)} вершин, {len(faces)} граней")
    
    def _is_triangular(self, mesh) -> bool:
        """Проверка, является ли меш треугольным"""
        if not hasattr(mesh, 'faces') or len(mesh.faces) == 0:
            return False
        return all(len(face) == 3 for face in mesh.faces)
    
    def _create_o3d_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray, float]:
        """Создание Open3D меша из вершин и граней"""
        # Нормализация
        center = np.mean(vertices, axis=0)
        vertices_centered = vertices - center
        max_distance = np.max(np.linalg.norm(vertices_centered, axis=1))
        
        if max_distance <= 1e-10:
            max_distance = 1.0
            logger.warning("Обнаружена вырожденная модель при нормализации")
        
        vertices_centered /= max_distance
        
        # Создание Open3D меша
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices_centered)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
        
        # Вычисление нормалей для улучшения ray casting
        mesh_o3d.compute_vertex_normals()
        mesh_o3d.compute_triangle_normals()
        
        return mesh_o3d, center, max_distance
    
    def _create_fallback_mesh(self) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray, float]:
        """Создание fallback меша при критических ошибках"""
        logger.warning("🔄 Создание fallback сферы")
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=10)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        center = np.zeros(3, dtype=np.float64)
        return mesh, center, 1.0
    
    def get_supported_formats(self) -> set:
        """Получение списка поддерживаемых форматов"""
        return self.SUPPORTED_FORMATS
    
    def clear_cache(self):
        """Очистка кэша загрузки"""
        self.load_cache.clear()
        logger.info("✅ Кэш загрузчика очищен")