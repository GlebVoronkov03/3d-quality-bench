# cache_manager.py
import pickle
import hashlib
import os
import zlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any, Dict
import logging
from dataclasses import is_dataclass, asdict
import json

logger = logging.getLogger(__name__)

class ComputationCache:
    """Менеджер кэширования результатов вычислений"""
    
    def __init__(self, cache_dir: str = ".pler_cache", ttl_hours: int = 24, max_cache_size: int = 1024):
        self.cache_dir = Path(cache_dir)
        self.ttl = timedelta(hours=ttl_hours)
        self.max_cache_size = max_cache_size  # в MB
        
        # Создаем директорию кэша
        self.cache_dir.mkdir(exist_ok=True)
        
        # Очищаем просроченные кэши при инициализации
        self._clean_expired_cache()
    
    def _get_cache_key(self, reference_path: str, distorted_path: str, num_rays: int, config_hash: str = "") -> str:
        """Генерация уникального ключа кэша на основе файлов и параметров"""
        file_stats = []
        
        for path in [reference_path, distorted_path]:
            path_obj = Path(path)
            if path_obj.exists():
                stat = path_obj.stat()
                file_stats.append(f"{path}_{stat.st_mtime}_{stat.st_size}")
            else:
                file_stats.append(f"{path}_missing")
        
        content = f"{'_'.join(file_stats)}_{num_rays}_{config_hash}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _get_cache_file_path(self, cache_key: str) -> Path:
        """Получение пути к файлу кэша"""
        return self.cache_dir / f"{cache_key}.pkl.gz"
    
    def _is_cache_valid(self, cache_file: Path) -> bool:
        """Проверка валидности кэша (TTL и размер)"""
        if not cache_file.exists():
            return False
        
        # Проверка TTL
        file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - file_time > self.ttl:
            return False
        
        # Проверка размера файла
        if cache_file.stat().st_size == 0:
            return False
            
        return True
    
    def _clean_expired_cache(self):
        """Очистка просроченных кэшей"""
        try:
            current_time = datetime.now()
            removed_count = 0
            
            for cache_file in self.cache_dir.glob("*.pkl.gz"):
                file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if current_time - file_time > self.ttl:
                    cache_file.unlink()
                    removed_count += 1
            
            if removed_count > 0:
                logger.info(f"🧹 Очищено {removed_count} просроченных кэшей")
                
            # Проверка общего размера кэша
            self._enforce_cache_size_limit()
            
        except Exception as e:
            logger.warning(f"Ошибка при очистке кэша: {e}")
    
    def _enforce_cache_size_limit(self):
        """Обеспечение ограничения размера кэша"""
        try:
            cache_files = list(self.cache_dir.glob("*.pkl.gz"))
            if not cache_files:
                return
            
            # Сортируем по времени модификации (старые сначала)
            cache_files.sort(key=lambda x: x.stat().st_mtime)
            
            total_size = sum(f.stat().st_size for f in cache_files) / 1024 / 1024  # в MB
            
            # Удаляем самые старые файлы пока не уложимся в лимит
            while total_size > self.max_cache_size and cache_files:
                oldest_file = cache_files.pop(0)
                file_size = oldest_file.stat().st_size / 1024 / 1024
                oldest_file.unlink()
                total_size -= file_size
                logger.debug(f"Удален кэш для соблюдения лимита: {oldest_file.name}")
                
        except Exception as e:
            logger.warning(f"Ошибка при ограничении размера кэша: {e}")
    
    def get_cached_result(self, reference_path: str, distorted_path: str, 
                         num_rays: int, config: Optional[Any] = None) -> Optional[Any]:
        """Получение закэшированного результата"""
        try:
            config_hash = self._get_config_hash(config)
            cache_key = self._get_cache_key(reference_path, distorted_path, num_rays, config_hash)
            cache_file = self._get_cache_file_path(cache_key)
            
            if self._is_cache_valid(cache_file):
                # Загрузка и декомпрессия
                with open(cache_file, 'rb') as f:
                    compressed_data = f.read()
                
                try:
                    decompressed_data = zlib.decompress(compressed_data)
                    result = pickle.loads(decompressed_data)
                    logger.info(f"✅ Используем закэшированный результат (ключ: {cache_key[:8]}...)")
                    return result
                except (zlib.error, pickle.UnpicklingError) as e:
                    logger.warning(f"Ошибка загрузки кэша, удаляем поврежденный файл: {e}")
                    cache_file.unlink()
                    return None
            
            return None
            
        except Exception as e:
            logger.warning(f"Ошибка получения кэша: {e}")
            return None
    
    def save_result(self, result: Any, reference_path: str, distorted_path: str, 
                   num_rays: int, config: Optional[Any] = None):
        """Сохранение результата в кэш"""
        try:
            config_hash = self._get_config_hash(config)
            cache_key = self._get_cache_key(reference_path, distorted_path, num_rays, config_hash)
            cache_file = self._get_cache_file_path(cache_key)
            
            # Сериализация и компрессия
            serialized_data = pickle.dumps(result)
            compressed_data = zlib.compress(serialized_data, level=6)
            
            # Сохранение
            with open(cache_file, 'wb') as f:
                f.write(compressed_data)
            
            logger.debug(f"💾 Результат сохранен в кэш (ключ: {cache_key[:8]}...)")
            
        except Exception as e:
            logger.warning(f"Ошибка сохранения в кэш: {e}")
    
    def _get_config_hash(self, config: Optional[Any]) -> str:
        """Генерация хеша конфигурации"""
        if config is None:
            return ""
        
        try:
            if is_dataclass(config):
                config_dict = asdict(config)
            elif hasattr(config, '__dict__'):
                config_dict = config.__dict__
            else:
                config_dict = str(config)
            
            config_str = json.dumps(config_dict, sort_keys=True, default=str)
            return hashlib.md5(config_str.encode()).hexdigest()
            
        except Exception as e:
            logger.warning(f"Ошибка генерации хеша конфигурации: {e}")
            return ""
    
    def clear_cache(self):
        """Полная очистка кэша"""
        try:
            cache_files = list(self.cache_dir.glob("*.pkl.gz"))
            for cache_file in cache_files:
                cache_file.unlink()
            
            logger.info(f"🧹 Очищено {len(cache_files)} файлов кэша")
            
        except Exception as e:
            logger.error(f"Ошибка очистки кэша: {e}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Получение информации о кэше"""
        try:
            cache_files = list(self.cache_dir.glob("*.pkl.gz"))
            total_size = sum(f.stat().st_size for f in cache_files) / 1024 / 1024  # MB
            oldest_file = min(cache_files, key=lambda x: x.stat().st_mtime) if cache_files else None
            newest_file = max(cache_files, key=lambda x: x.stat().st_mtime) if cache_files else None
            
            return {
                'total_files': len(cache_files),
                'total_size_mb': round(total_size, 2),
                'oldest_file': oldest_file.name if oldest_file else None,
                'newest_file': newest_file.name if newest_file else None,
                'cache_dir': str(self.cache_dir)
            }
        except Exception as e:
            logger.warning(f"Ошибка получения информации о кэше: {e}")
            return {}