# ray_analysis.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import sys
import os
from pler_metric_crash import PLERMetric

class PLERRayAnalysis:
    def __init__(self, reference_obj, distorted_obj):
        """
        Анализатор зависимости PLER от количества лучей
        
        Args:
            reference_obj (str): Путь к эталонной модели
            distorted_obj (str): Путь к искаженной модели
        """
        self.reference_obj = reference_obj
        self.distorted_obj = distorted_obj
        self.results = []
        
    def analyze_ray_range(self, min_rays=200, max_rays=200000, step=100, num_points=20):
        """
        Анализ метрики PLER в диапазоне количества лучей
        
        Args:
            min_rays (int): Минимальное количество лучей
            max_rays (int): Максимальное количество лучей  
            step (int): Шаг для линейного диапазона
            num_points (int): Количество точек для логарифмического диапазона
        """
        print("🔍 Начало анализа зависимости PLER от количества лучей...")
        print(f"📊 Диапазон: {min_rays} - {max_rays} лучей")
        
        # Создаем логарифмически распределенные точки для лучшего покрытия
        ray_counts = np.logspace(np.log10(min_rays), np.log10(max_rays), num=num_points, dtype=int)
        ray_counts = np.unique(ray_counts)  # Убираем дубликаты
        
        pler_calculator = PLERMetric()
        self.results = []
        
        for i, num_rays in enumerate(ray_counts):
            print(f"📏 Прогресс: {i+1}/{len(ray_counts)} | Лучи: {num_rays}")
            
            try:
                start_time = time.time()
                
                # Вычисляем PLER для текущего количества лучей
                result = pler_calculator.compute_pler(
                    self.reference_obj, 
                    self.distorted_obj, 
                    num_rays=num_rays
                )
                
                computation_time = time.time() - start_time
                
                # Сохраняем результаты
                result_data = {
                    'num_rays': num_rays,
                    'pler_db': result['PLER_dB'],
                    'mse': result['MSE'],
                    'mean_error': result['mean_error'],
                    'max_error': result['max_error'],
                    'computation_time': computation_time,
                    'time_per_ray': computation_time / num_rays
                }
                
                self.results.append(result_data)
                
                print(f"   PLER: {result['PLER_dB']:.2f} dB | "
                      f"Время: {computation_time:.2f}с | "
                      f"Время/луч: {computation_time/num_rays*1000:.3f}мс")
                      
            except Exception as e:
                print(f"❌ Ошибка для {num_rays} лучей: {e}")
                continue
        
        return self.results
    
    def find_optimal_rays(self, convergence_threshold=0.1, max_time_per_ray=0.001):
        """
        Поиск оптимального количества лучей
        
        Args:
            convergence_threshold (float): Порог сходимости (dB)
            max_time_per_ray (float): Максимальное допустимое время на луч (сек)
        """
        if not self.results:
            print("⚠️ Сначала выполните analyze_ray_range()")
            return None
        
        df = pd.DataFrame(self.results)
        
        # Находим точку сходимости (когда изменение PLER становится меньше порога)
        df['pler_diff'] = df['pler_db'].diff().abs()
        converged = df[df['pler_diff'] < convergence_threshold]
        
        if len(converged) > 0:
            # Первая точка сходимости
            optimal_by_convergence = converged.iloc[0]
        else:
            # Берем последнюю точку
            optimal_by_convergence = df.iloc[-1]
        
        # Учитываем время вычисления
        feasible = df[df['time_per_ray'] <= max_time_per_ray]
        if len(feasible) > 0:
            optimal_by_time = feasible.iloc[-1]  # Максимальное количество с допустимым временем
        else:
            optimal_by_time = df.iloc[0]
        
        # Компромиссное решение
        optimal_idx = min(optimal_by_convergence.name, optimal_by_time.name)
        optimal_result = df.loc[optimal_idx]
        
        print("\n🎯 РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ:")
        print(f"Оптимальное количество лучей: {int(optimal_result['num_rays'])}")
        print(f"PLER: {optimal_result['pler_db']:.2f} dB")
        print(f"Время вычисления: {optimal_result['computation_time']:.2f} сек")
        print(f"Время на луч: {optimal_result['time_per_ray']*1000:.3f} мс")
        
        return optimal_result
    
    def plot_analysis(self, save_path="ray_analysis_results.png"):
        """Построение графиков анализа"""
        if not self.results:
            print("⚠️ Нет данных для построения графиков")
            return
        
        df = pd.DataFrame(self.results)
        
        # Создаем subplot с 4 графиками
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # График 1: PLER vs Количество лучей
        ax1.plot(df['num_rays'], df['pler_db'], 'b-o', linewidth=2, markersize=4)
        ax1.set_xscale('log')
        ax1.set_xlabel('Количество лучей')
        ax1.set_ylabel('PLER (dB)')
        ax1.set_title('Зависимость PLER от количества лучей')
        ax1.grid(True, alpha=0.3)
        
        # График 2: Время вычисления vs Количество лучей
        ax2.plot(df['num_rays'], df['computation_time'], 'r-o', linewidth=2, markersize=4)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('Количество лучей')
        ax2.set_ylabel('Время вычисления (сек)')
        ax2.set_title('Зависимость времени от количества лучей')
        ax2.grid(True, alpha=0.3)
        
        # График 3: MSE vs Количество лучей
        ax3.plot(df['num_rays'], df['mse'], 'g-o', linewidth=2, markersize=4)
        ax3.set_xscale('log')
        ax3.set_xlabel('Количество лучей')
        ax3.set_ylabel('MSE')
        ax3.set_title('Зависимость MSE от количества лучей')
        ax3.grid(True, alpha=0.3)
        
        # График 4: Время на луч vs Количество лучей
        ax4.plot(df['num_rays'], df['time_per_ray']*1000, 'm-o', linewidth=2, markersize=4)
        ax4.set_xscale('log')
        ax4.set_xlabel('Количество лучей')
        ax4.set_ylabel('Время на луч (мс)')
        ax4.set_title('Эффективность вычислений')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Графики сохранены в {save_path}")
        
        # Дополнительный график: сходимость PLER
        plt.figure(figsize=(10, 6))
        plt.plot(df['num_rays'], df['pler_db'], 'b-o', linewidth=2, markersize=6)
        plt.xscale('log')
        plt.xlabel('Количество лучей')
        plt.ylabel('PLER (dB)')
        plt.title('Сходимость метрики PLER')
        plt.grid(True, alpha=0.3)
        
        # Показываем оптимальную точку
        optimal = self.find_optimal_rays()
        if optimal is not None:
            plt.axvline(x=optimal['num_rays'], color='red', linestyle='--', 
                       label=f'Оптимум: {int(optimal["num_rays"])} лучей')
            plt.legend()
        
        plt.savefig("pler_convergence.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig
    
    def save_results(self, filename="ray_analysis_results.csv"):
        """Сохранение результатов в CSV файл"""
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(filename, index=False)
            print(f"💾 Результаты сохранены в {filename}")
            return df
        else:
            print("⚠️ Нет данных для сохранения")
            return None

def main():
    """Основная функция для запуска анализа из командной строки"""
    if len(sys.argv) < 3:
        print("Использование: python ray_analysis.py <reference.obj> <distorted.obj>")
        print("Пример: python ray_analysis.py models/reference.obj models/distorted.obj")
        return
    
    reference_path = sys.argv[1]
    distorted_path = sys.argv[2]
    
    print("🚀 Запуск анализа оптимального количества лучей")
    print(f"Эталон: {reference_path}")
    print(f"Искаженная: {distorted_path}")
    print("-" * 50)
    
    # Создаем анализатор
    analyzer = PLERRayAnalysis(reference_path, distorted_path)
    
    try:
        # Выполняем анализ в диапазоне 100 - 20000 лучей
        results = analyzer.analyze_ray_range(
            min_rays=100,
            max_rays=20000, 
            step=100,
            num_points=25
        )
        
        # Находим оптимальное количество лучей
        optimal = analyzer.find_optimal_rays(
            convergence_threshold=0.5,  # dB
            max_time_per_ray=0.002      # сек
        )
        
        # Строим графики
        analyzer.plot_analysis()
        
        # Сохраняем результаты
        analyzer.save_results()
        
        print("\n✅ Анализ завершен!")
        if optimal is not None:
            print(f"🎯 Рекомендуемое количество лучей: {int(optimal['num_rays'])}")
            print(f"📈 Ожидаемый PLER: {optimal['pler_db']:.2f} dB")
            print(f"⏱️ Ожидаемое время: {optimal['computation_time']:.2f} сек")
            
    except Exception as e:
        print(f"❌ Ошибка при анализе: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()