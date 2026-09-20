import numpy as np
from typing import Dict, List, Optional
import json
from dataclasses import asdict
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    logger.warning("PyYAML не установлен, конфигурация будет использоваться по умолчанию")
    YAML_AVAILABLE = False

from pler_metric import PLERMetric, PLERConfig, PLERResult

try:
    from topology_analyzer import TopologyAnalyzer
    TOPOLOGY_AVAILABLE = True
except ImportError:
    logger.warning("TopologyAnalyzer не доступен")
    TOPOLOGY_AVAILABLE = False

class AdvancedPLERMetric(PLERMetric):
    """Расширенная версия метрики PLER с дополнительными функциями"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Инициализация logger для этого класса
        self.logger = logging.getLogger(__name__)
        
        if config_path and YAML_AVAILABLE:
            config = self._load_config(config_path)
        else:
            if config_path and not YAML_AVAILABLE:
                self.logger.warning("PyYAML не установлен, используется конфигурация по умолчанию")
            config = PLERConfig()
            
        super().__init__(config)
        
        # Инициализация анализатора топологии
        if TOPOLOGY_AVAILABLE:
            self.topology_analyzer = TopologyAnalyzer()
        else:
            self.topology_analyzer = None
            self.logger.warning("Топологический анализ отключен")
    
    def _load_config(self, config_path: str) -> PLERConfig:
        """Загрузка конфигурации из YAML файла"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            
            # Валидация конфигурации
            valid_keys = {field.name for field in PLERConfig.__dataclass_fields__.values()}
            filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
            
            self.logger.info(f"✅ Конфигурация загружена из {config_path}")
            return PLERConfig(**filtered_config)
            
        except Exception as e:
            self.logger.warning(f"Не удалось загрузить конфиг {config_path}: {e}")
            return PLERConfig()
    
    def save_config(self, config_path: str):
        """Сохранение конфигурации в YAML файл"""
        if not YAML_AVAILABLE:
            self.logger.error("PyYAML не установлен, сохранение конфигурации невозможно")
            return
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(asdict(self.config), f, default_flow_style=False, indent=2)
            self.logger.info(f"✅ Конфигурация сохранена в {config_path}")
        except Exception as e:
            self.logger.error(f"Ошибка сохранения конфигурации: {e}")
    
    def comprehensive_compare(self, reference_obj: str, distorted_obj: str,
                           num_rays: Optional[int] = None) -> Dict:
        """Комплексное сравнение моделей со всеми метриками"""
        
        self.logger.info("🎯 Запуск комплексного анализа...")
        
        # Базовая метрика PLER
        pler_result = self.compute_pler(reference_obj, distorted_obj, num_rays)
        
        # Топологический анализ
        topology_score = None
        if self.topology_analyzer:
            try:
                topology_score = self.topology_analyzer.compare_topology(reference_obj, distorted_obj)
                self.logger.info(f"📐 Топологическая оценка: {topology_score:.1%}")
            except Exception as e:
                self.logger.warning(f"Топологический анализ не удался: {e}")
        
        # Анализ сходимости
        convergence_valid = self.validate_convergence(reference_obj, distorted_obj)
        
        # Комбинированная оценка
        combined_score = self._compute_combined_score(pler_result, topology_score)
        
        result = {
            'pler_score': pler_result.pler_db,
            'topology_score': topology_score,
            'combined_score': combined_score,
            'convergence_valid': convergence_valid,
            'details': asdict(pler_result),
            'recommendation': self._generate_recommendation(combined_score, pler_result.pler_db)
        }
        
        self._print_comprehensive_results(result)
        return result
    
    def _compute_combined_score(self, pler_result: PLERResult, topology_score: float) -> float:
        """Вычисление комбинированной оценки"""
        # Нормализация PLER score (предполагаем, что >60 dB = отлично)
        pler_normalized = min(pler_result.pler_db / 60.0, 1.0)
        
        # Комбинация с весами
        if topology_score is not None:
            return 0.7 * pler_normalized + 0.3 * topology_score
        else:
            return pler_normalized
    
    def _generate_recommendation(self, combined_score: float, pler_db: float) -> str:
        """Генерация рекомендации на основе оценки"""
        if combined_score >= 0.9 or pler_db >= 60:
            return "✅ Отличное качество - изменения практически не заметны"
        elif combined_score >= 0.7 or pler_db >= 40:
            return "⚠️ Хорошее качество - незначительные изменения"
        elif combined_score >= 0.5 or pler_db >= 20:
            return "🔶 Удовлетворительное качество - заметные изменения"
        else:
            return "❌ Низкое качество - значительные искажения"
    
    def _print_comprehensive_results(self, result: Dict):
        """Красивый вывод комплексных результатов"""
        print("\n" + "="*60)
        print("🎯 КОМПЛЕКСНЫЙ АНАЛИЗ PLER:")
        print("="*60)
        print(f"PLER Score: {result['pler_score']:.2f} dB")
        
        if result['topology_score'] is not None:
            print(f"Topology Score: {result['topology_score']:.1%}")
        
        print(f"Combined Score: {result['combined_score']:.1%}")
        print(f"Convergence Valid: {'✅' if result['convergence_valid'] else '❌'}")
        print(f"\n📋 Рекомендация: {result['recommendation']}")
        
        # Детальная информация
        print("\n📊 Детали:")
        details = result['details']
        print(f"  • MSE: {details['mse']:.6f}")
        print(f"  • Средняя ошибка: {details['mean_error']:.3f}")
        print(f"  • Максимальная ошибка: {details['max_error']:.3f}")
        print(f"  • Количество лучей: {details['num_rays']}")
        print(f"  • Время вычисления: {details['computation_time']:.2f} сек")
        
        print("="*60)
    
    def batch_compare(self, reference_obj: str, distorted_objs: List[str]) -> Dict:
        """Пакетное сравнение эталонной модели с несколькими искаженными"""
        results = {}
        
        self.logger.info(f"🔄 Запуск пакетного сравнения с {len(distorted_objs)} моделями")
        
        for i, dist_obj in enumerate(distorted_objs, 1):
            try:
                self.logger.info(f"📦 Обработка {i}/{len(distorted_objs)}: {Path(dist_obj).name}")
                result = self.comprehensive_compare(reference_obj, dist_obj)
                results[dist_obj] = result
            except Exception as e:
                self.logger.error(f"❌ Ошибка сравнения {dist_obj}: {e}")
                results[dist_obj] = {'error': str(e)}
        
        # Сводка результатов
        self._print_batch_summary(results)
        return results
    
    def _print_batch_summary(self, results: Dict):
        """Вывод сводки пакетного анализа"""
        print("\n" + "="*50)
        print("📈 СВОДКА ПАКЕТНОГО АНАЛИЗА:")
        print("="*50)
        
        successful = [k for k, v in results.items() if 'error' not in v]
        errors = [k for k, v in results.items() if 'error' in v]
        
        print(f"✅ Успешно обработано: {len(successful)}")
        print(f"❌ Ошибок: {len(errors)}")
        
        if successful:
            print("\n🏆 Лучшие результаты:")
            best_results = sorted(
                [(k, v['pler_score']) for k, v in results.items() if 'error' not in v],
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            for model, score in best_results:
                print(f"  • {Path(model).name}: {score:.2f} dB")
        
        if errors:
            print(f"\n⚠️ Модели с ошибками: {len(errors)}")
            for model in errors[:3]:  # Показываем только первые 3
                print(f"  • {Path(model).name}")
    
    def create_simple_test_models(self):
        """Создание простых тестовых моделей для проверки работы"""
        import numpy as np
        
        def create_sphere_vertices_faces(radius=1.0, sectors=16, stacks=8):
            """Создание вершин и граней сферы"""
            vertices = []
            faces = []
            
            # Генерация вершин
            for i in range(stacks + 1):
                phi = np.pi * i / stacks
                for j in range(sectors):
                    theta = 2 * np.pi * j / sectors
                    x = radius * np.sin(phi) * np.cos(theta)
                    y = radius * np.sin(phi) * np.sin(theta)
                    z = radius * np.cos(phi)
                    vertices.append([x, y, z])
            
            # Генерация треугольников
            for i in range(stacks):
                for j in range(sectors):
                    first = i * sectors + j
                    second = first + sectors
                    next_j = (j + 1) % sectors
                    
                    first_next = i * sectors + next_j
                    second_next = second + next_j - j
                    
                    if i != 0:
                        faces.append([first, second, first_next])
                    if i != stacks - 1:
                        faces.append([first_next, second, second_next])
            
            return vertices, faces
        
        def save_obj(filename, vertices, faces):
            """Сохранение в OBJ формат"""
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("# Simple test mesh\n")
                for v in vertices:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                for face in faces:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        # Создание моделей
        vertices, faces = create_sphere_vertices_faces()
        
        # Эталонная модель
        save_obj("models/simple_reference.obj", vertices, faces)
        
        # Искаженная модель (немного сжатая)
        distorted_vertices = [[v[0]*0.9, v[1]*1.1, v[2]*0.95] for v in vertices]
        save_obj("models/simple_distorted.obj", distorted_vertices, faces)
        
        self.logger.info("✅ Созданы простые тестовые модели:")
        self.logger.info("  • models/simple_reference.obj")
        self.logger.info("  • models/simple_distorted.obj")

# Функция для запуска из командной строки
def main():
    """Основная функция для запуска расширенного анализа"""
    import sys
    from pathlib import Path
    
    if len(sys.argv) < 3:
        print("Использование: python pler_advanced.py <reference.obj> <distorted.obj>")
        print("Пример: python pler_advanced.py models/reference.obj models/distorted.obj")
        print("\nДополнительные команды:")
        print("  --create-test-models  Создать простые тестовые модели")
        return
    
    if sys.argv[1] == "--create-test-models":
        metric = AdvancedPLERMetric()
        metric.create_simple_test_models()
        return
    
    reference_path = sys.argv[1]
    distorted_path = sys.argv[2]
    
    # Проверка существования файлов
    if not Path(reference_path).exists():
        print(f"❌ Файл не найден: {reference_path}")
        return
    if not Path(distorted_path).exists():
        print(f"❌ Файл не найден: {distorted_path}")
        return
    
    metric = AdvancedPLERMetric()
    result = metric.comprehensive_compare(reference_path, distorted_path)

if __name__ == "__main__":
    main()
