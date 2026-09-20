# gpu_accelerator.py
import numpy as np
import logging
from typing import Optional, Tuple, Any
import time

logger = logging.getLogger(__name__)

class GPUAccelerator:
    """Базовый класс для GPU-ускорения вычислений"""
    
    def __init__(self):
        self.gpu_available = False
        self.gpu_lib = None
        self._initialize_gpu()
    
    def _initialize_gpu(self):
        """Инициализация GPU окружения"""
        try:
            # Пробуем импортировать CuPy для CUDA
            import cupy as cp
            self.gpu_lib = cp
            self.gpu_available = True
            logger.info("✅ CUDA доступна для GPU-ускорения")
            
        except ImportError:
            try:
                # Пробуем OpenCL через PyOpenCL
                import pyopencl as cl
                self.gpu_lib = cl
                self.gpu_available = True
                logger.info("✅ OpenCL доступен для GPU-ускорения")
                
            except ImportError:
                logger.warning("❌ CUDA и OpenCL недоступны, используем CPU")
                self.gpu_available = False
    
    def generate_rays_gpu(self, num_rays: int) -> Optional[np.ndarray]:
        """Генерация лучей на GPU (если доступно)"""
        if not self.gpu_available:
            return None
        
        try:
            start_time = time.time()
            
            if hasattr(self.gpu_lib, 'cupy'):
                # Используем CuPy (CUDA)
                rays_gpu = self._generate_rays_cupy(num_rays)
                rays_cpu = self.gpu_lib.asnumpy(rays_gpu)
                
            elif hasattr(self.gpu_lib, 'Context'):
                # Используем OpenCL
                rays_cpu = self._generate_rays_opencl(num_rays)
                
            else:
                return None
            
            computation_time = time.time() - start_time
            logger.debug(f"🚀 GPU генерация лучей: {num_rays} лучей за {computation_time:.3f}с")
            
            return rays_cpu
            
        except Exception as e:
            logger.warning(f"Ошибка GPU генерации лучей, используем CPU: {e}")
            return None
    
    def _generate_rays_cupy(self, num_rays: int) -> Any:
        """Генерация лучей через CuPy (CUDA)"""
        cp = self.gpu_lib
        
        # Фибоначчиева сфера на GPU
        indices = cp.arange(0, num_rays, dtype=float) + 0.5
        phi = cp.arccos(1 - 2 * indices / num_rays)
        theta = cp.pi * (1 + 5**0.5) * indices
        
        x = cp.cos(theta) * cp.sin(phi)
        y = cp.sin(theta) * cp.sin(phi)
        z = cp.cos(phi)
        
        return cp.stack([x, y, z], axis=1).astype(cp.float32)
    
    def _generate_rays_opencl(self, num_rays: int) -> np.ndarray:
        """Генерация лучей через OpenCL"""
        # Базовая реализация OpenCL (упрощенная)
        # В реальном проекте здесь был бы код OpenCL kernel
        logger.debug("OpenCL реализация требует дополнительной разработки")
        return None
    
    def ray_cast_gpu(self, mesh_data: Any, ray_directions: np.ndarray) -> Optional[np.ndarray]:
        """Ray casting на GPU (заглушка для будущей реализации)"""
        if not self.gpu_available:
            return None
        
        try:
            # TODO: Полная реализация GPU ray casting
            # Это потребует переноса меша на GPU и реализации ray-triangle intersection
            logger.debug("GPU ray casting в разработке...")
            return None
            
        except Exception as e:
            logger.warning(f"Ошибка GPU ray casting: {e}")
            return None
    
    def distance_computation_gpu(self, distances_ref: np.ndarray, distances_dist: np.ndarray) -> Optional[Tuple]:
        """Вычисление метрик на GPU"""
        if not self.gpu_available or len(distances_ref) == 0:
            return None
        
        try:
            start_time = time.time()
            
            if hasattr(self.gpu_lib, 'cupy'):
                # Перенос данных на GPU
                d_ref = self.gpu_lib.asarray(distances_ref.astype(np.float32))
                d_dist = self.gpu_lib.asarray(distances_dist.astype(np.float32))
                
                # Вычисления на GPU
                errors = self.gpu_lib.abs(d_ref - d_dist)
                mse = self.gpu_lib.mean((d_ref - d_dist) ** 2)
                mean_error = self.gpu_lib.mean(errors)
                max_error = self.gpu_lib.max(errors)
                
                # Перенос результатов обратно на CPU
                mse_cpu = float(self.gpu_lib.asnumpy(mse))
                mean_error_cpu = float(self.gpu_lib.asnumpy(mean_error))
                max_error_cpu = float(self.gpu_lib.asnumpy(max_error))
                
                computation_time = time.time() - start_time
                logger.debug(f"🚀 GPU вычисление метрик за {computation_time:.3f}с")
                
                return mse_cpu, mean_error_cpu, max_error_cpu
                
        except Exception as e:
            logger.warning(f"Ошибка GPU вычислений: {e}")
        
        return None
    
    def is_available(self) -> bool:
        """Проверка доступности GPU"""
        return self.gpu_available
    
    def get_gpu_info(self) -> dict:
        """Получение информации о GPU"""
        if not self.gpu_available:
            return {'available': False}
        
        try:
            if hasattr(self.gpu_lib, 'cupy'):
                # Информация о CUDA
                return {
                    'available': True,
                    'backend': 'CUDA',
                    'device_count': self.gpu_lib.cuda.runtime.getDeviceCount(),
                    'current_device': self.gpu_lib.cuda.runtime.getDevice()
                }
            elif hasattr(self.gpu_lib, 'Context'):
                # Информация об OpenCL
                platforms = self.gpu_lib.get_platforms()
                return {
                    'available': True,
                    'backend': 'OpenCL',
                    'platforms': len(platforms),
                    'devices': [len(platform.get_devices()) for platform in platforms]
                }
        except Exception as e:
            logger.warning(f"Ошибка получения информации о GPU: {e}")
        
        return {'available': False}