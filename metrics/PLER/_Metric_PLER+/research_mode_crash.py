# research_mode.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from pathlib import Path
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple
import logging
from perception_predictor import enhance_research_with_ml, PerceptionPredictor

# Настройка matplotlib для русских шрифтов
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

from pler_metric_crash import PLERMetric, PLERResult
from pler_advanced import AdvancedPLERMetric
from topology_analyzer import TopologyAnalyzer

class PLERResearchAnalyzer:
    """Анализатор для исследовательского режима PLER"""
    
    def __init__(self):
        self.basic_metric = PLERMetric()
        self.advanced_metric = AdvancedPLERMetric()
        self.topology_analyzer = TopologyAnalyzer()
        self.results = []
        
    def analyze_models_series(self, reference_path: str, distorted_pattern: str, num_models: int):
        """Анализ серии моделей"""
        print(f"🔍 Начало исследования для {num_models} моделей...")
        
        for i in range(1, num_models + 1):
            distorted_path = distorted_pattern.format(i)
            if not Path(distorted_path).exists():
                print(f"⚠️ Файл {distorted_path} не найден, пропускаем.")
                continue
                
            print(f"📊 Анализ модели {i}/{num_models}: {Path(distorted_path).name}")
            
            try:
                # Базовый анализ
                basic_result = self.basic_metric.compute_pler(reference_path, distorted_path)
                
                # Расширенный анализ
                advanced_result = self.advanced_metric.comprehensive_compare(reference_path, distorted_path)
                
                # Топологический анализ
                topology_score = self.topology_analyzer.compare_topology(reference_path, distorted_path)
                
                # Сбор полной статистики
                model_stats = self._collect_detailed_stats(
                    reference_path, distorted_path, basic_result, advanced_result, topology_score
                )
                
                self.results.append(model_stats)
                print(f"✅ Модель {i} проанализирована")
                
            except Exception as e:
                print(f"❌ Ошибка анализа модели {i}: {e}")
                continue
    
    def _collect_detailed_stats(self, ref_path: str, dist_path: str, 
                              basic_result: PLERResult, advanced_result: Dict,
                              topology_score: float) -> Dict:
        """Сбор детальной статистики для модели"""
        
        # Базовые метрики
        stats = {
            'model_name': Path(dist_path).name,
            'timestamp': datetime.now().isoformat(),
            'pler_db': basic_result.pler_db,
            'mse': basic_result.mse,
            'mean_error': basic_result.mean_error,
            'max_error': basic_result.max_error,
            'num_rays': basic_result.num_rays,
            'computation_time': basic_result.computation_time,
            'topology_score': topology_score,
            'combined_score': advanced_result.get('combined_score', 0),
            'convergence_valid': advanced_result.get('convergence_valid', False)
        }
        
        # Дополнительные статистические показатели
        ray_data = self._get_ray_distribution_data(ref_path, dist_path)
        stats.update(ray_data)
        
        return stats
    
    def _get_ray_distribution_data(self, ref_path: str, dist_path: str) -> Dict:
        """Получение данных о распределении лучей"""
        try:
            # Загрузка моделей для анализа распределения
            ref_mesh, _, _ = self.basic_metric._load_and_normalize_mesh(ref_path)
            dist_mesh, _, _ = self.basic_metric._load_and_normalize_mesh(dist_path)
            
            # Генерация лучей
            num_rays = 1000  # Фиксированное количество для сравнения
            rays = self.basic_metric._generate_uniform_rays(num_rays)
            
            # Ray casting
            ref_distances = self.basic_metric._ray_cast_mesh(ref_mesh, rays)
            dist_distances = self.basic_metric._ray_cast_mesh(dist_mesh, rays)
            
            # Вычисление L-значений
            L_ref = 1.0 - ref_distances
            L_dist = 1.0 - dist_distances
            
            # Статистика распределения
            errors = np.abs(L_ref - L_dist)
            
            return {
                'ray_errors_mean': np.mean(errors),
                'ray_errors_std': np.std(errors),
                'ray_errors_skew': stats.skew(errors),
                'ray_errors_kurtosis': stats.kurtosis(errors),
                'ray_correlation': np.corrcoef(L_ref, L_dist)[0, 1],
                'human_perception_score': self._calculate_human_perception_score(errors)
            }
        except Exception as e:
            print(f"⚠️ Ошибка анализа распределения лучей: {e}")
            return {}
    
    def _calculate_human_perception_score(self, errors: np.ndarray) -> float:
        """Расчет оценки человеческого восприятия на основе ошибок"""
        # Основано на психофизических моделях (закон Вебера-Фехнера)
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        # Логарифмическая зависимость от ошибки (более чувствительны к малым ошибкам)
        if mean_error > 0:
            perception_score = 10 * np.log10(1.0 / mean_error)
        else:
            perception_score = 100.0
            
        # Учет вариативности ошибок
        if std_error > 0:
            perception_score *= (1.0 - 0.1 * np.log1p(std_error))
            
        return max(0, min(100, perception_score))
    
    def generate_comprehensive_report(self):
        """Генерация комплексного отчета"""
        if not self.results:
            print("❌ Нет данных для отчета")
            return
            
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*80)
        print("🎯 КОМПЛЕКСНЫЙ ИССЛЕДОВАТЕЛЬСКИЙ ОТЧЕТ PLER")
        print("="*80)
        
        # Базовая статистика
        self._print_basic_statistics(df)
        
        # Расширенная статистика
        self._print_advanced_statistics(df)
        
        # Корреляционный анализ
        self._print_correlation_analysis(df)
        
        # Визуализация
        self._generate_research_plots(df)
        
        # Сохранение отчета
        self._save_research_report(df)
    
    def _print_basic_statistics(self, df: pd.DataFrame):
        """Вывод базовой статистики"""
        print("\n📊 БАЗОВАЯ СТАТИСТИКА:")
        print(f"Количество моделей: {len(df)}")
        print(f"PLER Score - Среднее: {df['pler_db'].mean():.2f} dB")
        print(f"PLER Score - Стандартное отклонение: {df['pler_db'].std():.2f} dB")
        print(f"PLER Score - Диапазон: [{df['pler_db'].min():.2f}, {df['pler_db'].max():.2f}] dB")
        print(f"MSE - Среднее: {df['mse'].mean():.6f}")
        print(f"MSE - Стандартное отклонение: {df['mse'].std():.6f}")
        
    def _print_advanced_statistics(self, df: pd.DataFrame):
        """Вывод расширенной статистики"""
        print("\n📈 РАСШИРЕННАЯ СТАТИСТИКА:")
        
        # Вероятностные распределения
        for col in ['pler_db', 'mse', 'mean_error', 'combined_score']:
            if col in df.columns:
                data = df[col].dropna()
                if len(data) > 1:
                    print(f"\n{col.upper()}:")
                    print(f"  Математическое ожидание: {data.mean():.4f}")
                    print(f"  Дисперсия: {data.var():.4f}")
                    print(f"  Асимметрия: {stats.skew(data):.4f}")
                    print(f"  Эксцесс: {stats.kurtosis(data):.4f}")
                    print(f"  95% доверительный интервал: {stats.t.interval(0.95, len(data)-1, loc=data.mean(), scale=stats.sem(data))}")
        
        # Дополнительные метрики
        if 'ray_errors_std' in df.columns:
            print(f"\nОшибки лучей - CV (Коэффициент вариации): {(df['ray_errors_std'] / df['ray_errors_mean']).mean()*100:.2f}%")
        
    def _print_correlation_analysis(self, df: pd.DataFrame):
        """Корреляционный анализ"""
        print("\n🔗 КОРРЕЛЯЦИОННЫЙ АНАЛИЗ:")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        # Корреляция с PLER
        if 'pler_db' in correlation_matrix.columns:
            pler_correlations = correlation_matrix['pler_db'].sort_values(ascending=False)
            print("Корреляция с PLER Score:")
            for metric, corr in pler_correlations.items():
                if metric != 'pler_db' and abs(corr) > 0.1:
                    print(f"  {metric}: {corr:.3f}")
    
    def _print_ml_analysis(self, df: pd.DataFrame):
        """Вывод ML анализа"""
        if 'ml_perception_prediction' not in df.columns:
            return
        
        print("\n🤖 ML АНАЛИЗ ВОСПРИЯТИЯ:")
        print(f"Среднее предсказание восприятия: {df['ml_perception_prediction'].mean():.1f}")
        print(f"Стандартное отклонение: {df['ml_perception_prediction'].std():.1f}")
        
        # Корреляция с другими метриками
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        ml_correlations = df[numeric_cols].corr()['ml_perception_prediction'].sort_values(ascending=False)
        
        print("\nКорреляция с ML предсказанием:")
        for metric, corr in ml_correlations.items():
            if metric != 'ml_perception_prediction' and abs(corr) > 0.3:
                print(f"  {metric}: {corr:.3f}")
    
    def _generate_research_plots(self, df: pd.DataFrame):
        """Генерация исследовательских графиков"""
        print("\n📊 Генерация графиков...")
        
        # 1. Распределение PLER scores
        plt.figure(figsize=(12, 8))
        
        # 1.1 Гистограмма распределения PLER
        plt.subplot(2, 3, 1)
        plt.hist(df['pler_db'], bins=15, alpha=0.7, edgecolor='black')
        plt.xlabel('PLER (dB)')
        plt.ylabel('Частота')
        plt.title('Распределение PLER Scores')
        plt.grid(True, alpha=0.3)
        
        # 1.2 Box plot всех метрик
        plt.subplot(2, 3, 2)
        plot_data = df[['pler_db', 'mse', 'mean_error', 'combined_score']].copy()
        plot_data = (plot_data - plot_data.mean()) / plot_data.std()  # Нормализация
        plt.boxplot(plot_data.values, labels=plot_data.columns)
        plt.xticks(rotation=45)
        plt.title('Сравнение метрик (нормализованные)')
        plt.grid(True, alpha=0.3)
        
        # 1.3 Q-Q plot для PLER
        plt.subplot(2, 3, 3)
        stats.probplot(df['pler_db'], dist="norm", plot=plt)
        plt.title('Q-Q Plot PLER Scores')
        plt.grid(True, alpha=0.3)
        
        # 1.4 Зависимость ошибок от человеческого восприятия
        if 'human_perception_score' in df.columns:
            plt.subplot(2, 3, 4)
            plt.scatter(df['mean_error'], df['human_perception_score'], alpha=0.6)
            plt.xlabel('Средняя ошибка')
            plt.ylabel('Оценка восприятия')
            plt.title('Зависимость восприятия от ошибок')
            
            # Линия тренда
            z = np.polyfit(df['mean_error'], df['human_perception_score'], 1)
            p = np.poly1d(z)
            plt.plot(df['mean_error'], p(df['mean_error']), "r--", alpha=0.8)
            plt.grid(True, alpha=0.3)
        
        # 1.5 Heatmap корреляций
        plt.subplot(2, 3, 5)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns)
        plt.title('Матрица корреляций')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # 1.6 Временная диаграмма вычислений
        plt.subplot(2, 3, 6)
        plt.plot(df['computation_time'], 'o-', alpha=0.7)
        plt.xlabel('Номер модели')
        plt.ylabel('Время вычисления (сек)')
        plt.title('Производительность вычислений')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('research_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Дополнительные исследовательские графики
        self._generate_additional_plots(df)
    
    def _generate_additional_plots(self, df: pd.DataFrame):
        """Генерация дополнительных исследовательских графиков"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 2.1 Распределение ошибок по моделям
        if 'ray_errors_std' in df.columns:
            axes[0, 0].errorbar(range(len(df)), df['ray_errors_mean'], 
                               yerr=df['ray_errors_std'], fmt='o', capsize=5)
            axes[0, 0].set_xlabel('Модель')
            axes[0, 0].set_ylabel('Средняя ошибка лучей')
            axes[0, 0].set_title('Распределение ошибок лучей по моделям')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2.2 Сравнение топологических оценок
        if 'topology_score' in df.columns:
            axes[0, 1].scatter(df['topology_score'], df['pler_db'], alpha=0.6)
            axes[0, 1].set_xlabel('Топологическая оценка')
            axes[0, 1].set_ylabel('PLER (dB)')
            axes[0, 1].set_title('Зависимость PLER от топологии')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 2.3 Кумулятивное распределение PLER
        axes[1, 0].hist(df['pler_db'], bins=20, cumulative=True, density=True, 
                       alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('PLER (dB)')
        axes[1, 0].set_ylabel('Кумулятивная вероятность')
        axes[1, 0].set_title('Кумулятивное распределение PLER')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 2.4 3D scatter plot (если достаточно данных)
        if len(df) >= 3 and all(col in df.columns for col in ['pler_db', 'mse', 'combined_score']):
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(2, 2, 4, projection='3d')
            ax.scatter(df['pler_db'], df['mse'], df['combined_score'], alpha=0.6)
            ax.set_xlabel('PLER (dB)')
            ax.set_ylabel('MSE')
            ax.set_zlabel('Combined Score')
            ax.set_title('3D распределение метрик')
        
        plt.tight_layout()
        plt.savefig('research_additional_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _save_research_report(self, df: pd.DataFrame):
        """Сохранение отчета исследования"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_models': len(df),
            'summary_statistics': {
                'pler_db': {
                    'mean': float(df['pler_db'].mean()),
                    'std': float(df['pler_db'].std()),
                    'min': float(df['pler_db'].min()),
                    'max': float(df['pler_db'].max())
                },
                'computation_time': {
                    'total': float(df['computation_time'].sum()),
                    'average': float(df['computation_time'].mean())
                }
            },
            'models': df.to_dict('records')
        }
        
        with open('research_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        df.to_csv('research_data.csv', index=False, encoding='utf-8')
        print(f"💾 Отчет сохранен в research_report.json и research_data.csv")
    def enhance_with_ml_predictions(self):
        """Улучшение данных с помощью ML предсказаний"""
        if not self.results:
            logger.warning("Нет данных для ML обработки")
            return
        
        # Преобразуем результаты в DataFrame
        df = pd.DataFrame(self.results)
        
        # Применяем ML улучшения
        enhanced_df = enhance_research_with_ml(df)
        
        # Обновляем результаты
        self.results = enhanced_df.to_dict('records')
        
        logger.info("✅ Данные улучшены с помощью ML предсказаний")

def main():
    """Основная функция исследовательского режима"""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='PLER Research Mode')
    parser.add_argument('--num-models', type=int, required=True,
                       help='Количество искаженных моделей для анализа')
    parser.add_argument('--reference', type=str, default='models/reference.obj',
                       help='Путь к эталонной модели')
    parser.add_argument('--pattern', type=str, default='models/distorted_{}.obj',
                       help='Шаблон путей к искаженным моделям')
    
    args = parser.parse_args()
    
    analyzer = PLERResearchAnalyzer()
    
    try:
        # Анализ серии моделей
        analyzer.analyze_models_series(
            args.reference,
            args.pattern,
            args.num_models
        )
        
        # Генерация комплексного отчета
        analyzer.generate_comprehensive_report()
        
        print("\n✅ Исследование завершено!")
        
    except Exception as e:
        print(f"❌ Ошибка исследования: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()