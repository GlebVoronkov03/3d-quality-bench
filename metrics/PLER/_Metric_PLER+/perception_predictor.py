# perception_predictor.py
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class PerceptionFeatures:
    """Признаки для прогнозирования восприятия"""
    # Геометрические признаки
    pler_score: float
    mse: float
    mean_error: float
    max_error: float
    ray_errors_std: float
    ray_errors_skew: float
    ray_errors_kurtosis: float
    
    # Топологические признаки
    topology_score: float
    vertex_count: int
    face_count: int
    edge_count: int
    euler_characteristic: int
    genus: float
    
    # Статистические признаки
    human_perception_score: float
    computation_time: float
    ray_correlation: float
    
    # Дополнительные геометрические признаки
    surface_area: float = 0.0
    volume: float = 0.0
    compactness: float = 0.0
    bounding_box_ratio: float = 0.0

class PerceptionPredictor:
    """ML модель для прогнозирования человеческого восприятия качества 3D моделей"""
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self.training_history = {}
        
        # Инициализация модели
        self._initialize_model()
    
    def _initialize_model(self):
        """Инициализация ML модели"""
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == 'linear':
            self.model = LinearRegression()
        elif self.model_type == 'svr':
            self.model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        else:
            raise ValueError(f"Неизвестный тип модели: {self.model_type}")
        
        logger.info(f"✅ Инициализирована ML модель: {self.model_type}")
    
    def extract_features(self, research_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Извлечение признаков из исследовательских данных"""
        features_list = []
        targets = []
        
        for _, row in research_data.iterrows():
            try:
                features = self._extract_single_sample_features(row)
                features_list.append(features)
                
                # Целевая переменная - комбинация PLER и человеческого восприятия
                target = self._calculate_target_value(row)
                targets.append(target)
                
            except Exception as e:
                logger.warning(f"Ошибка извлечения признаков: {e}")
                continue
        
        return np.array(features_list), np.array(targets)
    
    def _extract_single_sample_features(self, row: pd.Series) -> List[float]:
        """Извлечение признаков из одного образца"""
        features = []
        
        # Базовые метрики PLER
        features.extend([
            row.get('pler_db', 0),
            row.get('mse', 0),
            row.get('mean_error', 0),
            row.get('max_error', 0)
        ])
        
        # Статистические признаки лучей
        features.extend([
            row.get('ray_errors_std', 0),
            row.get('ray_errors_skew', 0),
            row.get('ray_errors_kurtosis', 0),
            row.get('ray_correlation', 0)
        ])
        
        # Топологические признаки
        features.extend([
            row.get('topology_score', 0),
            row.get('human_perception_score', 0)
        ])
        
        # Временные характеристики
        features.append(row.get('computation_time', 0))
        
        # Дополнительные вычисляемые признаки
        features.extend(self._compute_derived_features(row))
        
        return features
    
    def _compute_derived_features(self, row: pd.Series) -> List[float]:
        """Вычисление производных признаков"""
        derived = []
        
        # Относительные ошибки
        pler_db = row.get('pler_db', 1)
        mse = row.get('mse', 0.001)
        
        # Логарифмические преобразования
        derived.append(np.log1p(pler_db) if pler_db > 0 else 0)
        derived.append(np.log1p(mse) if mse > 0 else 0)
        
        # Соотношения метрик
        if mse > 0:
            derived.append(pler_db / mse)
        else:
            derived.append(pler_db)
        
        # Нормализованные ошибки
        mean_error = row.get('mean_error', 0)
        max_error = row.get('max_error', 0)
        
        if max_error > 0:
            derived.append(mean_error / max_error)
        else:
            derived.append(0)
        
        return derived
    
    def _calculate_target_value(self, row: pd.Series) -> float:
        """Вычисление целевой переменной для обучения"""
        # Комбинированная оценка на основе PLER и человеческого восприятия
        pler_score = row.get('pler_db', 0)
        perception_score = row.get('human_perception_score', 0)
        topology_score = row.get('topology_score', 0)
        
        # Нормализация PLER (0-100)
        pler_normalized = min(pler_score / 60.0 * 100, 100) if pler_score > 0 else 0
        
        # Комбинированная целевая переменная
        target = (
            0.6 * pler_normalized + 
            0.3 * perception_score + 
            0.1 * topology_score * 100
        )
        
        return max(0, min(100, target))
    
    def train(self, research_data: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
        """Обучение ML модели"""
        try:
            # Извлечение признаков и целей
            X, y = self.extract_features(research_data)
            
            if len(X) < 10:
                raise ValueError(f"Недостаточно данных для обучения. Доступно: {len(X)} samples")
            
            # Разделение на train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Масштабирование признаков
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Обучение модели
            self.model.fit(X_train_scaled, y_train)
            
            # Предсказание и оценка
            y_pred = self.model.predict(X_test_scaled)
            
            # Метрики качества
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Кросс-валидация
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, 
                                      cv=min(5, len(X_train)), scoring='r2')
            
            self.is_trained = True
            self.feature_names = self._get_feature_names()
            
            # Сохранение истории обучения
            self.training_history = {
                'model_type': self.model_type,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'mse': mse,
                'mae': mae,
                'r2_score': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'feature_importance': self._get_feature_importance()
            }
            
            logger.info(f"✅ Модель обучена. R²: {r2:.3f}, MAE: {mae:.2f}")
            
            # Визуализация результатов
            self._plot_training_results(y_test, y_pred)
            
            return self.training_history
            
        except Exception as e:
            logger.error(f"❌ Ошибка обучения модели: {e}")
            raise
    
    def predict(self, features: PerceptionFeatures) -> float:
        """Предсказание оценки восприятия для новых данных"""
        if not self.is_trained:
            raise RuntimeError("Модель не обучена. Сначала вызовите train()")
        
        try:
            # Преобразование признаков в вектор
            feature_vector = self._features_to_vector(features)
            
            # Масштабирование
            feature_vector_scaled = self.scaler.transform([feature_vector])
            
            # Предсказание
            prediction = self.model.predict(feature_vector_scaled)[0]
            
            return max(0, min(100, prediction))
            
        except Exception as e:
            logger.error(f"Ошибка предсказания: {e}")
            return 50.0  # Возвращаем нейтральное значение при ошибке
    
    def _features_to_vector(self, features: PerceptionFeatures) -> List[float]:
        """Преобразование объекта признаков в вектор"""
        vector = [
            features.pler_score,
            features.mse,
            features.mean_error,
            features.max_error,
            features.ray_errors_std,
            features.ray_errors_skew,
            features.ray_errors_kurtosis,
            features.ray_correlation,
            features.topology_score,
            features.human_perception_score,
            features.computation_time
        ]
        
        # Добавляем производные признаки
        vector.extend([
            np.log1p(features.pler_score) if features.pler_score > 0 else 0,
            np.log1p(features.mse) if features.mse > 0 else 0,
            features.pler_score / features.mse if features.mse > 0 else features.pler_score,
            features.mean_error / features.max_error if features.max_error > 0 else 0
        ])
        
        return vector
    
    def _get_feature_names(self) -> List[str]:
        """Получение названий признаков"""
        base_features = [
            'pler_score', 'mse', 'mean_error', 'max_error',
            'ray_errors_std', 'ray_errors_skew', 'ray_errors_kurtosis', 'ray_correlation',
            'topology_score', 'human_perception_score', 'computation_time'
        ]
        
        derived_features = [
            'log_pler', 'log_mse', 'pler_mse_ratio', 'error_ratio'
        ]
        
        return base_features + derived_features
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Получение важности признаков"""
        if not hasattr(self.model, 'feature_importances_'):
            return {}
        
        importance_dict = {}
        for name, importance in zip(self.feature_names, self.model.feature_importances_):
            importance_dict[name] = float(importance)
        
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def _plot_training_results(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Визуализация результатов обучения"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Фактические vs Предсказанные значения
            axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
            axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            axes[0, 0].set_xlabel('Фактические значения')
            axes[0, 0].set_ylabel('Предсказанные значения')
            axes[0, 0].set_title('Фактические vs Предсказанные значения')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Ошибки предсказания
            errors = y_true - y_pred
            axes[0, 1].hist(errors, bins=20, alpha=0.7, edgecolor='black')
            axes[0, 1].axvline(x=0, color='r', linestyle='--')
            axes[0, 1].set_xlabel('Ошибка предсказания')
            axes[0, 1].set_ylabel('Частота')
            axes[0, 1].set_title('Распределение ошибок предсказания')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Важность признаков
            if hasattr(self.model, 'feature_importances_'):
                importance = self._get_feature_importance()
                features = list(importance.keys())[:10]  # Топ-10 признаков
                importances = list(importance.values())[:10]
                
                axes[1, 0].barh(features, importances)
                axes[1, 0].set_xlabel('Важность признака')
                axes[1, 0].set_title('Топ-10 важных признаков')
            
            # 4. Q-Q plot ошибок
            stats.probplot(errors, dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title('Q-Q Plot ошибок предсказания')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('ml_training_results.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("📊 Графики обучения сохранены в ml_training_results.png")
            
        except Exception as e:
            logger.warning(f"Ошибка визуализации результатов обучения: {e}")
    
    def save_model(self, filepath: str):
        """Сохранение обученной модели"""
        if not self.is_trained:
            raise RuntimeError("Модель не обучена")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'model_type': self.model_type
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"💾 Модель сохранена в {filepath}")
    
    def load_model(self, filepath: str):
        """Загрузка обученной модели"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.training_history = model_data['training_history']
        self.model_type = model_data['model_type']
        self.is_trained = True
        
        logger.info(f"📥 Модель загружена из {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Получение информации о модели"""
        if not self.is_trained:
            return {'is_trained': False}
        
        info = {
            'is_trained': True,
            'model_type': self.model_type,
            'training_samples': self.training_history.get('training_samples', 0),
            'r2_score': self.training_history.get('r2_score', 0),
            'feature_importance': self.training_history.get('feature_importance', {})
        }
        
        return info

class SyntheticDataGenerator:
    """Генератор синтетических данных для начального обучения"""
    
    def __init__(self, num_samples: int = 1000):
        self.num_samples = num_samples
    
    def generate_data(self) -> pd.DataFrame:
        """Генерация синтетических данных для обучения"""
        data = []
        
        for i in range(self.num_samples):
            sample = {
                'model_name': f'synthetic_{i}',
                'pler_db': np.random.uniform(10, 80),
                'mse': np.random.uniform(0.001, 0.1),
                'mean_error': np.random.uniform(0.01, 0.5),
                'max_error': np.random.uniform(0.1, 1.0),
                'ray_errors_std': np.random.uniform(0.01, 0.2),
                'ray_errors_skew': np.random.uniform(-2, 2),
                'ray_errors_kurtosis': np.random.uniform(-1, 5),
                'ray_correlation': np.random.uniform(0.5, 1.0),
                'topology_score': np.random.uniform(0.5, 1.0),
                'human_perception_score': np.random.uniform(20, 95),
                'computation_time': np.random.uniform(0.5, 10.0)
            }
            data.append(sample)
        
        return pd.DataFrame(data)

# Интеграция с исследовательским режимом
def enhance_research_with_ml(research_data: pd.DataFrame) -> pd.DataFrame:
    """Улучшение исследовательских данных с помощью ML"""
    
    # Создаем и обучаем модель на доступных данных
    predictor = PerceptionPredictor(model_type='random_forest')
    
    # Если данных мало, используем синтетические данные для начального обучения
    if len(research_data) < 20:
        logger.info("📊 Данных мало, генерируем синтетические данные для обучения...")
        synthetic_generator = SyntheticDataGenerator(num_samples=100)
        synthetic_data = synthetic_generator.generate_data()
        
        # Объединяем реальные и синтетические данные
        combined_data = pd.concat([research_data, synthetic_data], ignore_index=True)
        training_data = combined_data
    else:
        training_data = research_data
    
    # Обучаем модель
    try:
        training_results = predictor.train(training_data)
        logger.info(f"🎯 ML модель обучена. R²: {training_results['r2_score']:.3f}")
        
        # Добавляем предсказания к исходным данным
        enhanced_data = research_data.copy()
        predictions = []
        
        for _, row in research_data.iterrows():
            try:
                # Создаем объект признаков
                features = PerceptionFeatures(
                    pler_score=row.get('pler_db', 0),
                    mse=row.get('mse', 0),
                    mean_error=row.get('mean_error', 0),
                    max_error=row.get('max_error', 0),
                    ray_errors_std=row.get('ray_errors_std', 0),
                    ray_errors_skew=row.get('ray_errors_skew', 0),
                    ray_errors_kurtosis=row.get('ray_errors_kurtosis', 0),
                    topology_score=row.get('topology_score', 0),
                    vertex_count=0,
                    face_count=0,
                    edge_count=0,
                    euler_characteristic=0,
                    genus=0,
                    human_perception_score=row.get('human_perception_score', 0),
                    computation_time=row.get('computation_time', 0),
                    ray_correlation=row.get('ray_correlation', 0)
                )
                
                prediction = predictor.predict(features)
                predictions.append(prediction)
                
            except Exception as e:
                logger.warning(f"Ошибка предсказания для строки: {e}")
                predictions.append(50.0)  # Нейтральное значение
        
        enhanced_data['ml_perception_prediction'] = predictions
        
        # Сохраняем модель для будущего использования
        predictor.save_model('perception_predictor.pkl')
        
        return enhanced_data
        
    except Exception as e:
        logger.error(f"❌ Ошибка ML обработки: {e}")
        return research_data

def main():
    """Демонстрация работы ML модуля"""
    # Генерация синтетических данных для демонстрации
    generator = SyntheticDataGenerator(num_samples=200)
    synthetic_data = generator.generate_data()
    
    # Улучшение данных с помощью ML
    enhanced_data = enhance_research_with_ml(synthetic_data)
    
    print("✅ ML модуль успешно протестирован")
    print(f"📊 Обработано образцов: {len(enhanced_data)}")
    
    if 'ml_perception_prediction' in enhanced_data.columns:
        print(f"🎯 Добавлены ML предсказания восприятия")
        print(f"   Среднее предсказание: {enhanced_data['ml_perception_prediction'].mean():.1f}")
        print(f"   Стандартное отклонение: {enhanced_data['ml_perception_prediction'].std():.1f}")

if __name__ == "__main__":
    main()