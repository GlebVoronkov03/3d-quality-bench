# sphere_analysis.py
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
import time
from datetime import datetime
from pathlib import Path
import argparse
import json

class SphereAnalyzer:
    """Анализатор сфер из OBJ файлов"""
    
    def __init__(self):
        self.results = []
        
    def analyze_series(self, reference_path: str, pattern: str, num_models: int):
        """Анализ серии моделей"""
        print(f"🔍 Анализ {num_models} моделей...")
        
        if not Path(reference_path).exists():
            print(f"❌ Эталон не найден: {reference_path}")
            return
            
        try:
            ref_mesh = trimesh.load_mesh(reference_path)
            print(f"✅ Эталон загружен: {len(ref_mesh.vertices)} вершин")
        except Exception as e:
            print(f"❌ Ошибка загрузки эталона: {e}")
            return

        for i in range(1, num_models + 1):
            model_path = pattern.format(i)
            if not Path(model_path).exists():
                print(f"⚠️ Пропуск: {model_path}")
                continue
                
            print(f"📊 Модель {i}/{num_models}: {Path(model_path).name}")
            
            try:
                start_time = time.time()
                test_mesh = trimesh.load_mesh(model_path)
                
                # Вычисление всех метрик
                stats_data = self._compute_all_metrics(ref_mesh, test_mesh, model_path, i)
                stats_data['computation_time'] = time.time() - start_time
                
                self.results.append(stats_data)
                self._save_single_plot(stats_data, i)
                print(f"✅ Модель {i} проанализирована")
                
            except Exception as e:
                print(f"❌ Ошибка: {e}")
                continue

    def _compute_all_metrics(self, ref_mesh, test_mesh, model_path, model_id):
        """Вычисление всех метрик для одной модели"""
        # Базовые характеристики
        stats_data = {
            'model_id': model_id,
            'model_name': Path(model_path).name,
            'vertices_count': len(test_mesh.vertices),
            'faces_count': len(test_mesh.faces),
            'timestamp': datetime.now().isoformat()
        }
        
        # MCM анализ
        mcm_results = self._compute_mcm_metrics(ref_mesh, test_mesh)
        stats_data.update(mcm_results)
        
        # Геометрические метрики
        geo_results = self._compute_geometric_metrics(ref_mesh, test_mesh)
        stats_data.update(geo_results)
        
        # Статистика ошибок
        if 'mcm_distances' in mcm_results:
            stat_results = self._compute_error_stats(mcm_results['mcm_distances'])
            stats_data.update(stat_results)
        
        return stats_data

    def _compute_mcm_metrics(self, ref_mesh, test_mesh):
        """Вычисление MCM метрик"""
        try:
            ref_vertices = np.array(ref_mesh.vertices)
            test_vertices = np.array(test_mesh.vertices)
            
            ref_kdtree = cKDTree(ref_vertices)
            distances, _ = ref_kdtree.query(test_vertices)
            
            # MCM метрики
            AAD = np.mean(distances)
            SSD = np.std(distances) if len(distances) > 1 else 0
            
            # Относительные искажения размеров
            ref_size = ref_mesh.bounds[1] - ref_mesh.bounds[0]
            test_size = test_mesh.bounds[1] - test_mesh.bounds[0]
            ADRO = np.mean(np.abs((test_size / ref_size - 1) * 100))
            
            return {
                'MCM_AAD': AAD,
                'MCM_SSD': SSD,
                'MCM_ADRO': ADRO,
                'mcm_distances': distances
            }
        except Exception as e:
            print(f"Ошибка MCM: {e}")
            return {'MCM_AAD': 0, 'MCM_SSD': 0, 'MCM_ADRO': 0}

    def _compute_geometric_metrics(self, ref_mesh, test_mesh):
        """Вычисление геометрических метрик"""
        try:
            ref_vol, test_vol = ref_mesh.volume, test_mesh.volume
            ref_area, test_area = ref_mesh.area, test_mesh.area
            
            vol_error = abs(test_vol - ref_vol) / ref_vol * 100 if ref_vol > 0 else 100
            area_error = abs(test_area - ref_area) / ref_area * 100 if ref_area > 0 else 100
            
            return {
                'volume_error': vol_error,
                'area_error': area_error,
                'ref_volume': ref_vol,
                'test_volume': test_vol,
                'ref_area': ref_area,
                'test_area': test_area
            }
        except Exception as e:
            print(f"Ошибка геометрии: {e}")
            return {'volume_error': 0, 'area_error': 0}

    def _compute_error_stats(self, distances):
        """Статистический анализ ошибок"""
        if len(distances) == 0:
            return {}
            
        return {
            'error_mean': np.mean(distances),
            'error_std': np.std(distances),
            'error_max': np.max(distances),
            'error_min': np.min(distances),
            'error_median': np.median(distances)
        }

    def _save_single_plot(self, stats, model_id):
        """Сохранение графика распределения ошибок для одной модели"""
        if 'mcm_distances' not in stats or len(stats['mcm_distances']) == 0:
            return
            
        distances = stats['mcm_distances']
        model_name = stats['model_name'].replace('.obj', '')
        
        plt.figure(figsize=(10, 6))
        
        # Гистограмма с нормальным распределением
        n, bins, _ = plt.hist(distances, bins=50, alpha=0.7, density=True, 
                             edgecolor='black', label='Распределение ошибок')
        
        # Нормальное распределение
        if len(distances) > 1:
            mu, sigma = np.mean(distances), np.std(distances)
            x = np.linspace(bins[0], bins[-1], 100)
            y = stats.norm.pdf(x, mu, sigma)
            plt.plot(x, y, 'r-', linewidth=2, label=f'Нормальное распределение\nμ={mu:.6f}, σ={sigma:.6f}')
        
        plt.xlabel('Величина ошибки')
        plt.ylabel('Плотность вероятности')
        plt.title(f'Распределение MCM ошибок: {model_name}\n'
                 f'Вершин: {stats["vertices_count"]}, MCM_AAD: {stats["MCM_AAD"]:.6f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'model_{model_id:02d}_{model_name}_distribution.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()

    def generate_report(self):
        """Генерация отчета и сохранение результатов"""
        if not self.results:
            print("❌ Нет данных для отчета")
            return
            
        df = pd.DataFrame(self.results)
        
        # Сохранение CSV
        df.to_csv('sphere_analysis_results.csv', index=False, encoding='utf-8')
        print("💾 Результаты сохранены в sphere_analysis_results.csv")
        
        # Статистика
        print("\n📊 СВОДНАЯ СТАТИСТИКА:")
        print(f"Моделей проанализировано: {len(df)}")
        print(f"MCM_AAD: {df['MCM_AAD'].mean():.6f} ± {df['MCM_AAD'].std():.6f}")
        print(f"MCM_SSD: {df['MCM_SSD'].mean():.6f} ± {df['MCM_SSD'].std():.6f}")
        print(f"Ошибка объема: {df['volume_error'].mean():.2f}% ± {df['volume_error'].std():.2f}%")
        
        # Корреляции
        if len(df) > 1:
            corr = df['vertices_count'].corr(df['MCM_AAD'])
            print(f"Корреляция вершин-MCM_AAD: {corr:.3f}")
        
        # Графики трендов
        self._plot_trends(df)

    def _plot_trends(self, df):
        """Построение графиков трендов"""
        if len(df) < 2:
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # MCM метрики vs детализация
        ax1.semilogx(df['vertices_count'], df['MCM_AAD'], 'bo-', label='MCM_AAD')
        ax1.semilogx(df['vertices_count'], df['MCM_SSD'], 'ro-', label='MCM_SSD')
        ax1.set_xlabel('Количество вершин')
        ax1.set_ylabel('MCM метрики')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title('MCM метрики vs Детализация')
        
        # Геометрические ошибки
        ax2.semilogx(df['vertices_count'], df['volume_error'], 'go-', label='Ошибка объема')
        ax2.semilogx(df['vertices_count'], df['area_error'], 'mo-', label='Ошибка площади')
        ax2.set_xlabel('Количество вершин')
        ax2.set_ylabel('Ошибка, %')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Геометрические ошибки vs Детализация')
        
        # Время вычислений
        ax3.semilogx(df['vertices_count'], df['computation_time'], 'co-')
        ax3.set_xlabel('Количество вершин')
        ax3.set_ylabel('Время вычисления, сек')
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Производительность анализа')
        
        # ADRO
        ax4.bar(range(len(df)), df['MCM_ADRO'], alpha=0.7)
        ax4.set_xlabel('Модель')
        ax4.set_ylabel('ADRO, %')
        ax4.set_xticks(range(len(df)))
        ax4.set_xticklabels([f'M{i+1}' for i in range(len(df))], rotation=45)
        ax4.grid(True, alpha=0.3)
        ax4.set_title('Относительные искажения по моделям')
        
        plt.tight_layout()
        plt.savefig('sphere_analysis_trends.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("📈 Графики трендов сохранены в sphere_analysis_trends.png")

def main():
    parser = argparse.ArgumentParser(description='Анализ качества 3D сфер')
    parser.add_argument('--num-models', type=int, required=True, help='Количество тестовых моделей')
    parser.add_argument('--reference', type=str, required=True, help='Путь к эталонной модели')
    parser.add_argument('--pattern', type=str, required=True, help='Шаблон путей к тестовым моделям')
    
    args = parser.parse_args()
    
    analyzer = SphereAnalyzer()
    
    try:
        analyzer.analyze_series(args.reference, args.pattern, args.num_models)
        analyzer.generate_report()
        print("\n✅ Анализ завершен!")
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")

if __name__ == "__main__":
    main()