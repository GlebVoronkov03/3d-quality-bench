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
import trimesh
from scipy.spatial import cKDTree

# Настройка matplotlib для русских шрифтов
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

from pler_metric import PLERMetric, PLERResult
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
                
                # MCM анализ (новый метод)
                mcm_results = self._compute_MCM_metrics(reference_path, distorted_path)
                
                # Сбор полной статистики
                model_stats = self._collect_detailed_stats(
                    reference_path, distorted_path, basic_result, advanced_result, topology_score, mcm_results
                )
                
                self.results.append(model_stats)
                print(f"✅ Модель {i} проанализирована")
                
            except Exception as e:
                print(f"❌ Ошибка анализа модели {i}: {e}")
                continue
    
    def _compute_MCM_metrics(self, ref_path: str, dist_path: str) -> Dict:
        """Вычисление MCM метрик (Mesh Comparison Method)"""
        try:
            # Загрузка моделей
            ref_mesh = trimesh.load_mesh(ref_path)
            dist_mesh = trimesh.load_mesh(dist_path)
            
            # Получение вершин моделей
            ref_vertices = np.array(ref_mesh.vertices)
            dist_vertices = np.array(dist_mesh.vertices)
            
            # Построение KD-дерева для эталонной модели
            ref_kdtree = cKDTree(ref_vertices)
            
            # Нахождение ближайших соседей для каждой точки искаженной модели
            distances, indices = ref_kdtree.query(dist_vertices)
            
            # MCM метрики
            n = len(distances)
            
            # 1. AAD (Arithmetic Average of Deviation)
            AAD = np.mean(distances)
            
            # 2. SSD (Sample Standard Deviation)
            SSD = np.sqrt(np.sum((distances - AAD)**2) / (n - 1))
            
            # 3. ADRO (Average Distortions Relative to Original)
            # Вычисление максимальных размеров по осям
            ref_bounds = ref_mesh.bounds
            dist_bounds = dist_mesh.bounds
            
            ref_sizes = ref_bounds[1] - ref_bounds[0]  # [size_x, size_y, size_z]
            dist_sizes = dist_bounds[1] - dist_bounds[0]
            
            # Относительные искажения по осям в процентах
            distortions = (dist_sizes / ref_sizes - 1) * 100
            ADRO = np.mean(np.abs(distortions))
            
            return {
                'AAD': AAD,
                'SSD': SSD,
                'ADRO': ADRO,
                'mcm_distances': distances,
                'ref_size_x': ref_sizes[0],
                'ref_size_y': ref_sizes[1],
                'ref_size_z': ref_sizes[2],
                'dist_size_x': dist_sizes[0],
                'dist_size_y': dist_sizes[1],
                'dist_size_z': dist_sizes[2]
            }
            
        except Exception as e:
            print(f"⚠️ Ошибка MCM анализа: {e}")
            return {'AAD': np.nan, 'SSD': np.nan, 'ADRO': np.nan, 'mcm_distances': np.array([])}
    
    def _collect_detailed_stats(self, ref_path: str, dist_path: str, 
                              basic_result: PLERResult, advanced_result: Dict,
                              topology_score: float, mcm_results: Dict) -> Dict:
        """Сбор детальной статистики для модели"""
        
        # Базовые метрики (переименованные согласно требованиям)
        stats = {
            'model_name': Path(dist_path).name,
            'timestamp': datetime.now().isoformat(),
            'pler_db': basic_result.pler_db,
            'AAD': basic_result.mean_error,  # Переименовано
            'mse': basic_result.mse,
            'max_error': basic_result.max_error,
            'num_rays': basic_result.num_rays,
            'computation_time': basic_result.computation_time,
            'topology_score': topology_score,
            'combined_score': advanced_result.get('combined_score', 0),
            'convergence_valid': advanced_result.get('convergence_valid', False)
        }
        
        # Новые MCM метрики
        stats.update({
            'MCM_AAD': mcm_results.get('AAD', np.nan),      # Среднее арифметическое отклонений по MCM
            'MCM_SSD': mcm_results.get('SSD', np.nan),      # Стандартное отклонение по MCM
            'MCM_ADRO': mcm_results.get('ADRO', np.nan),    # Средние искажения относительно оригинала
        })
        
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
        print(f"MCM_AAD - Среднее: {df['MCM_AAD'].mean():.6f} мм")
        print(f"MCM_SSD - Среднее: {df['MCM_SSD'].mean():.6f} мм")
        print(f"MCM_ADRO - Среднее: {df['MCM_ADRO'].mean():.2f}%")
        
    def _print_advanced_statistics(self, df: pd.DataFrame):
        """Вывод расширенной статистики"""
        print("\n📈 РАСШИРЕННАЯ СТАТИСТИКА:")
        
        # Вероятностные распределения
        for col in ['pler_db', 'mse', 'AAD', 'combined_score', 'MCM_AAD', 'MCM_SSD']:
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
    
    def _generate_research_plots(self, df: pd.DataFrame):
        """Генерация исследовательских графиков для научной публикации"""
        print("\n📊 Генерация графиков для научной публикации...")
        
        # Создаем фигуру с профессиональным оформлением
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Тренды основных метрик по последовательности моделей
        plt.subplot(3, 4, 1)
        models_order = range(1, len(df) + 1)
        
        plt.plot(models_order, df['pler_db'], 'o-', linewidth=2, markersize=6, label='PLER (dB)')
        plt.plot(models_order, df['MCM_AAD'] * 1000, 's-', linewidth=2, markersize=6, label='MCM_AAD (×10³)')
        plt.xlabel('Номер модели')
        plt.ylabel('Метрики качества')
        plt.title('Тренды метрик качества по моделям')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Сравнение MCM метрик
        plt.subplot(3, 4, 2)
        x_pos = np.arange(len(df))
        width = 0.35
        
        plt.bar(x_pos - width/2, df['MCM_AAD'] * 1000, width, label='MCM_AAD (×10³)')
        plt.bar(x_pos + width/2, df['MCM_SSD'] * 1000, width, label='MCM_SSD (×10³)')
        plt.xlabel('Модели')
        plt.ylabel('Отклонения (×10³)')
        plt.title('Сравнение MCM метрик')
        plt.legend()
        plt.xticks(x_pos, [f'M{i+1}' for i in range(len(df))], rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 3. Корреляционная матрица (heatmap)
        plt.subplot(3, 4, 3)
        key_metrics = ['pler_db', 'MCM_AAD', 'MCM_SSD', 'MCM_ADRO', 'topology_score', 'combined_score']
        corr_matrix = df[key_metrics].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.2f',
                   square=True, cbar_kws={'shrink': 0.8})
        plt.title('Матрица корреляций ключевых метрик')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # 4. Распределение PLER scores с нормальным распределением
        plt.subplot(3, 4, 4)
        pler_data = df['pler_db'].dropna()
        plt.hist(pler_data, bins=12, density=True, alpha=0.7, edgecolor='black', label='Данные')
        
        # Fit normal distribution
        mu, std = stats.norm.fit(pler_data)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        plt.plot(x, p, 'r-', linewidth=2, label=f'Нормальное распределение\nμ={mu:.2f}, σ={std:.2f}')
        plt.xlabel('PLER (dB)')
        plt.ylabel('Плотность вероятности')
        plt.title('Распределение PLER Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Scatter plot: MCM_AAD vs PLER с линией регрессии
        plt.subplot(3, 4, 5)
        plt.plot(df['MCM_AAD'] * 1000, df['pler_db'], 'o', alpha=0.7)
        
        # Линия регрессии
        z = np.polyfit(df['MCM_AAD'] * 1000, df['pler_db'], 1)
        p = np.poly1d(z)
        plt.plot(df['MCM_AAD'] * 1000, p(df['MCM_AAD'] * 1000), "r--", alpha=0.8, 
                label=f'R² = {np.corrcoef(df["MCM_AAD"], df["pler_db"])[0,1]**2:.3f}')
        
        plt.xlabel('MCM_AAD (×10³)')
        plt.ylabel('PLER (dB)')
        plt.title('Зависимость PLER от MCM_AAD')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. Box plot всех ключевых метрик
        plt.subplot(3, 4, 6)
        plot_data = df[['pler_db', 'MCM_AAD', 'MCM_SSD', 'MCM_ADRO']].copy()
        # Масштабирование для лучшей визуализации
        plot_data['MCM_AAD'] = plot_data['MCM_AAD'] * 1000
        plot_data['MCM_SSD'] = plot_data['MCM_SSD'] * 1000
        plt.boxplot(plot_data.values, labels=['PLER', 'MCM_AAD\n(×10³)', 'MCM_SSD\n(×10³)', 'MCM_ADRO'])
        plt.title('Распределение ключевых метрик')
        plt.grid(True, alpha=0.3)
        
        # 7. Кумулятивное распределение MCM_AAD
        plt.subplot(3, 4, 7)
        sorted_aad = np.sort(df['MCM_AAD'] * 1000)
        yvals = np.arange(len(sorted_aad)) / float(len(sorted_aad) - 1)
        plt.plot(sorted_aad, yvals, 'b-', linewidth=2)
        plt.xlabel('MCM_AAD (×10³)')
        plt.ylabel('Кумулятивная вероятность')
        plt.title('Кумулятивное распределение MCM_AAD')
        plt.grid(True, alpha=0.3)
        
        # 8. Временные характеристики
        plt.subplot(3, 4, 8)
        plt.plot(models_order, df['computation_time'], 'o-', linewidth=2, markersize=6)
        plt.xlabel('Номер модели')
        plt.ylabel('Время вычисления (сек)')
        plt.title('Производительность вычислений')
        plt.grid(True, alpha=0.3)
        
        # 9. MCM распределение для выбранных моделей
        plt.subplot(3, 4, 9)
        if len(self.results) >= 3:
            # Берем лучшую, среднюю и худшую модели по PLER
            best_idx = df['pler_db'].idxmax()
            worst_idx = df['pler_db'].idxmin()
            mid_idx = len(df) // 2
            
            for idx, label in zip([best_idx, mid_idx, worst_idx], ['Лучшая', 'Средняя', 'Худшая']):
                if 'mcm_distances' in self.results[idx]:
                    distances = self.results[idx]['mcm_distances']
                    if len(distances) > 0:
                        plt.hist(distances * 1000, bins=50, alpha=0.6, density=True, 
                                label=f'{label} (AAD: {self.results[idx]["MCM_AAD"]*1000:.1f})')
            
            plt.xlabel('Отклонение (×10³)')
            plt.ylabel('Плотность вероятности')
            plt.title('Распределение MCM отклонений')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 10. Сравнение размеров моделей (ADRO)
        plt.subplot(3, 4, 10)
        if all(col in df.columns for col in ['dist_size_x', 'dist_size_y', 'dist_size_z']):
            sizes = ['X', 'Y', 'Z']
            ref_sizes = [df['ref_size_x'].iloc[0], df['ref_size_y'].iloc[0], df['ref_size_z'].iloc[0]]
            dist_sizes_mean = [df['dist_size_x'].mean(), df['dist_size_y'].mean(), df['dist_size_z'].mean()]
            
            x = np.arange(len(sizes))
            plt.bar(x - 0.2, ref_sizes, 0.4, label='Эталон')
            plt.bar(x + 0.2, dist_sizes_mean, 0.4, label='Искаженные (среднее)')
            plt.xlabel('Оси')
            plt.ylabel('Размер')
            plt.title('Сравнение размеров моделей')
            plt.xticks(x, sizes)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 11. Radar chart для комплексной оценки
        plt.subplot(3, 4, 11)
        self._create_radar_chart(df)
        
        # 12. Q-Q plot для MCM_AAD
        plt.subplot(3, 4, 12)
        stats.probplot(df['MCM_AAD'], dist="norm", plot=plt)
        plt.title('Q-Q Plot MCM_AAD')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('research_scientific_publication.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Дополнительные специализированные графики
        self._generate_MCM_distribution_plots(df)
    
    def _create_radar_chart(self, df: pd.DataFrame):
        """Создание radar chart для комплексной оценки"""
        # Нормализация метрик для radar chart (0-1, где 1 лучше)
        metrics = ['pler_db', 'MCM_AAD', 'topology_score', 'combined_score']
        normalized = {}
        
        for metric in metrics:
            if metric == 'MCM_AAD':
                # Для MCM_AAD инвертируем (меньше = лучше)
                values = 1 - (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
            else:
                values = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
            normalized[metric] = values
        
        # Углы для осей radar chart
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Замыкаем круг
        
        # Средние значения
        mean_values = [normalized[metric].mean() for metric in metrics]
        mean_values += mean_values[:1]
        
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
        ax.plot(angles, mean_values, 'o-', linewidth=2, label='Среднее по всем моделям')
        ax.fill(angles, mean_values, alpha=0.25)
        
        # Добавляем метки
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(['PLER', 'MCM_AAD\n(инверт.)', 'Топология', 'Комбинир.'])
        ax.set_ylim(0, 1)
        ax.set_title('Radar Chart качества моделей')
        ax.grid(True)
        ax.legend(loc='upper right')
    
    def _generate_MCM_distribution_plots(self, df: pd.DataFrame):
        """Генерация отдельных графиков распределения MCM"""
        print("\n📊 Генерация MCM распределений...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Нормальное распределение MCM_AAD
        axes[0, 0].hist(df['MCM_AAD'] * 1000, bins=15, density=True, alpha=0.7, 
                       edgecolor='black', label='Данные MCM_AAD')
        
        # Fit normal distribution
        mu, std = stats.norm.fit(df['MCM_AAD'] * 1000)
        x = np.linspace(axes[0, 0].get_xlim()[0], axes[0, 0].get_xlim()[1], 100)
        p = stats.norm.pdf(x, mu, std)
        axes[0, 0].plot(x, p, 'r-', linewidth=2, 
                       label=f'Нормальное распределение\nμ={mu:.2f}, σ={std:.2f}')
        
        axes[0, 0].set_xlabel('MCM_AAD (×10³)')
        axes[0, 0].set_ylabel('Плотность вероятности')
        axes[0, 0].set_title('Распределение MCM_AAD (похоже на Гауссовское)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Сравнение распределений отклонений для разных моделей
        if len(self.results) >= 5:
            models_to_plot = min(5, len(self.results))
            for i in range(models_to_plot):
                if 'mcm_distances' in self.results[i]:
                    distances = self.results[i]['mcm_distances'] * 1000
                    if len(distances) > 0:
                        axes[0, 1].hist(distances, bins=30, alpha=0.5, 
                                      label=f'Модель {i+1}', density=True)
            
            axes[0, 1].set_xlabel('Отклонение (×10³)')
            axes[0, 1].set_ylabel('Плотность вероятности')
            axes[0, 1].set_title('Сравнение MCM распределений по моделям')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Зависимость MCM метрик от номера модели
        models_order = range(1, len(df) + 1)
        axes[1, 0].plot(models_order, df['MCM_AAD'] * 1000, 'o-', 
                       linewidth=2, markersize=6, label='MCM_AAD (×10³)')
        axes[1, 0].plot(models_order, df['MCM_SSD'] * 1000, 's-', 
                       linewidth=2, markersize=6, label='MCM_SSD (×10³)')
        axes[1, 0].set_xlabel('Номер модели')
        axes[1, 0].set_ylabel('Отклонения (×10³)')
        axes[1, 0].set_title('Динамика MCM метрик по моделям')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 3D scatter plot для основных метрик
        if len(df) >= 3:
            from mpl_toolkits.mplot3d import Axes3D
            ax_3d = fig.add_subplot(2, 2, 4, projection='3d')
            
            scatter = ax_3d.scatter(df['MCM_AAD'] * 1000, df['MCM_SSD'] * 1000, 
                                   df['pler_db'], c=df['pler_db'], 
                                   cmap='viridis', s=50, alpha=0.7)
            
            ax_3d.set_xlabel('MCM_AAD (×10³)')
            ax_3d.set_ylabel('MCM_SSD (×10³)')
            ax_3d.set_zlabel('PLER (dB)')
            ax_3d.set_title('3D пространство метрик качества')
            plt.colorbar(scatter, ax=ax_3d, label='PLER (dB)')
        
        plt.tight_layout()
        plt.savefig('research_MCM_analysis.png', dpi=300, bbox_inches='tight')
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
                'MCM_AAD': {
                    'mean': float(df['MCM_AAD'].mean()),
                    'std': float(df['MCM_AAD'].std()),
                    'min': float(df['MCM_AAD'].min()),
                    'max': float(df['MCM_AAD'].max())
                },
                'MCM_SSD': {
                    'mean': float(df['MCM_SSD'].mean()),
                    'std': float(df['MCM_SSD'].std()),
                    'min': float(df['MCM_SSD'].min()),
                    'max': float(df['MCM_SSD'].max())
                },
                'MCM_ADRO': {
                    'mean': float(df['MCM_ADRO'].mean()),
                    'std': float(df['MCM_ADRO'].std()),
                    'min': float(df['MCM_ADRO'].min()),
                    'max': float(df['MCM_ADRO'].max())
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
                       help='Шаблон путей к искаженных моделям')
    
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