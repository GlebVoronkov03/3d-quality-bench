# Создание виртуального окружения

python -m venv pler_env

# Активация (Windows)

pler_env\Scripts\activate

# Деактивация текущего окружения (если активно)

deactivate

# Удаление проблемного окружения

rmdir /s pler_env

# Установка из requirements.txt

pip install -r requirements.txt

# Или установка вручную

pip install numpy>=1.21.0
pip install open3d>=0.15.1
pip install trimesh>=3.9.0
pip install scipy>=1.7.0
pip install matplotlib>=3.4.0
pip install pandas>=1.3.0
pip install PyYAML>=6.0
pip install networkx>=2.6.0
pip install scikit-learn>=1.0.0
pip install tqdm>=4.62.0
pip install pytest>=6.2.5

# Проверка основных модулей

python -c "import open3d as o3d; print('✅ Open3D установлен')"
python -c "import trimesh; print('✅ Trimesh установлен')"
python -c "import yaml; print('✅ PyYAML установлен')"
python -c "import networkx as nx; print('✅ NetworkX установлен')"

# Создание простых тестовых моделей

python pler_advanced.py --create-test-models

# Быстрая оценка с автоматическими настройками

python pler_metric.py models/simple_reference.obj models/simple_distorted.obj

# Комплексный анализ с топологией

python pler_advanced.py models/simple_reference.obj models/simple_distorted.obj

# Оценка с фиксированным количеством лучей

python pler_metric.py models/reference.obj models/distorted.obj 5000

# Анализ оптимального количества лучей

python ray_analysis.py <reference.obj> <distorted.obj>

# Анализ серии моделей (10 искаженных моделей)

python research*mode.py --num-models 10 --reference models/my_ref.obj --pattern "models/variant*{}.obj"

# последняя консоль

PS C:\_Metric*PLER+> python -m venv pler_env
PS C:\_Metric_PLER+> pler_env\Scripts\activate
(pler_env) PS C:\_Metric_PLER+> pip install -r requirements.txt
Requirement already satisfied: numpy>=1.21.0 in c:\_metric_pler+\pler_env\lib\site-packages (from -r requirements.txt (line 1)) (2.2.6)
Requirement already satisfied: open3d>=0.15.1 in c:\_metric_pler+\pler_env\lib\site-packages (from -r requirements.txt (line 2)) (0.19.0)
Requirement already satisfied: trimesh>=3.9.0 in c:\_metric_pler+\pler_env\lib\site-packages (from -r requirements.txt (line 3)) (4.9.0)
Requirement already satisfied: scipy>=1.7.0 in c:\_metric_pler+\pler_env\lib\site-packages (from -r requirements.txt (line 4)) (1.15.3)
Requirement already satisfied: matplotlib>=3.4.0 in c:\_metric_pler+\pler_env\lib\site-packages (from -r requirements.txt (line 5)) (3.10.7)
Requirement already satisfied: pandas>=1.3.0 in c:\_metric_pler+\pler_env\lib\site-packages (from -r requirements.txt (line 6)) (2.3.3)
Requirement already satisfied: PyYAML>=6.0 in c:\_metric_pler+\pler_env\lib\site-packages (from -r requirements.txt (line 7)) (6.0.3)
Requirement already satisfied: networkx>=2.6.0 in c:\_metric_pler+\pler_env\lib\site-packages (from -r requirements.txt (line 8)) (3.4.2)
Requirement already satisfied: scikit-learn>=1.0.0 in c:\_metric_pler+\pler_env\lib\site-packages (from -r requirements.txt (line 9)) (1.7.2)
Requirement already satisfied: tqdm>=4.62.0 in c:\_metric_pler+\pler_env\lib\site-packages (from -r requirements.txt (line 10)) (4.67.1)
Requirement already satisfied: pytest>=6.2.5 in c:\_metric_pler+\pler_env\lib\site-packages (from -r requirements.txt (line 11)) (9.0.1)
Requirement already satisfied: seaborn>=0.13.2 in c:\_metric_pler+\pler_env\lib\site-packages (from -r requirements.txt (line 12)) (0.13.2)
Requirement already satisfied: dash>=2.6.0 in c:\_metric_pler+\pler_env\lib\site-packages (from open3d>=0.15.1->-r requirements.txt (line 2)) (3.3.0)
Requirement already satisfied: werkzeug>=3.0.0 in c:\_metric_pler+\pler_env\lib\site-packages (from open3d>=0.15.1->-r requirements.txt (line 2)) (3.1.3)
Requirement already satisfied: flask>=3.0.0 in c:\_metric_pler+\pler_env\lib\site-packages (from open3d>=0.15.1->-r requirements.txt (line 2)) (3.1.2)
Requirement already satisfied: nbformat>=5.7.0 in c:\_metric_pler+\pler_env\lib\site-packages (from open3d>=0.15.1->-r requirements.txt (line 2)) (5.7.0)
Requirement already satisfied: configargparse in c:\_metric_pler+\pler_env\lib\site-packages (from open3d>=0.15.1->-r requirements.txt (line 2)) (1.7.1)
Requirement already satisfied: ipywidgets>=8.0.4 in c:\_metric_pler+\pler_env\lib\site-packages (from open3d>=0.15.1->-r requirements.txt (line 2)) (8.1.8)
Requirement already satisfied: contourpy>=1.0.1 in c:\_metric_pler+\pler_env\lib\site-packages (from matplotlib>=3.4.0->-r requirements.txt (line 5)) (1.3.2)
Requirement already satisfied: cycler>=0.10 in c:\_metric_pler+\pler_env\lib\site-packages (from matplotlib>=3.4.0->-r requirements.txt (line 5)) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in c:\_metric_pler+\pler_env\lib\site-packages (from matplotlib>=3.4.0->-r requirements.txt (line 5)) (4.60.1)
Requirement already satisfied: kiwisolver>=1.3.1 in c:\_metric_pler+\pler_env\lib\site-packages (from matplotlib>=3.4.0->-r requirements.txt (line 5)) (1.4.9)
Requirement already satisfied: packaging>=20.0 in c:\_metric_pler+\pler_env\lib\site-packages (from matplotlib>=3.4.0->-r requirements.txt (line 5)) (25.0)
Requirement already satisfied: pillow>=8 in c:\_metric_pler+\pler_env\lib\site-packages (from matplotlib>=3.4.0->-r requirements.txt (line 5)) (12.0.0)
Requirement already satisfied: pyparsing>=3 in c:\_metric_pler+\pler_env\lib\site-packages (from matplotlib>=3.4.0->-r requirements.txt (line 5)) (3.2.5)
Requirement already satisfied: python-dateutil>=2.7 in c:\_metric_pler+\pler_env\lib\site-packages (from matplotlib>=3.4.0->-r requirements.txt (line 5)) (2.9.0.post0)
Requirement already satisfied: pytz>=2020.1 in c:\_metric_pler+\pler_env\lib\site-packages (from pandas>=1.3.0->-r requirements.txt (line 6)) (2025.2)
Requirement already satisfied: tzdata>=2022.7 in c:\_metric_pler+\pler_env\lib\site-packages (from pandas>=1.3.0->-r requirements.txt (line 6)) (2025.2)
Requirement already satisfied: joblib>=1.2.0 in c:\_metric_pler+\pler_env\lib\site-packages (from scikit-learn>=1.0.0->-r requirements.txt (line 9)) (1.5.2)
Requirement already satisfied: threadpoolctl>=3.1.0 in c:\_metric_pler+\pler_env\lib\site-packages (from scikit-learn>=1.0.0->-r requirements.txt (line 9)) (3.6.0)
Requirement already satisfied: colorama in c:\_metric_pler+\pler_env\lib\site-packages (from tqdm>=4.62.0->-r requirements.txt (line 10)) (0.4.6)
Requirement already satisfied: exceptiongroup>=1 in c:\_metric_pler+\pler_env\lib\site-packages (from pytest>=6.2.5->-r requirements.txt (line 11)) (1.3.0)
Requirement already satisfied: iniconfig>=1.0.1 in c:\_metric_pler+\pler_env\lib\site-packages (from pytest>=6.2.5->-r requirements.txt (line 11)) (2.3.0)
Requirement already satisfied: pluggy<2,>=1.5 in c:\_metric_pler+\pler_env\lib\site-packages (from pytest>=6.2.5->-r requirements.txt (line 11)) (1.6.0)
Requirement already satisfied: pygments>=2.7.2 in c:\_metric_pler+\pler_env\lib\site-packages (from pytest>=6.2.5->-r requirements.txt (line 11)) (2.19.2)
Requirement already satisfied: tomli>=1 in c:\_metric_pler+\pler_env\lib\site-packages (from pytest>=6.2.5->-r requirements.txt (line 11)) (2.3.0)
Requirement already satisfied: plotly>=5.0.0 in c:\_metric_pler+\pler_env\lib\site-packages (from dash>=2.6.0->open3d>=0.15.1->-r requirements.txt (line 2)) (6.4.0)
Requirement already satisfied: importlib-metadata in c:\_metric_pler+\pler_env\lib\site-packages (from dash>=2.6.0->open3d>=0.15.1->-r requirements.txt (line 2)) (8.7.0)
Requirement already satisfied: typing_extensions>=4.1.1 in c:\_metric_pler+\pler_env\lib\site-packages (from dash>=2.6.0->open3d>=0.15.1->-r requirements.txt (line 2)) (4.15.0)
Requirement already satisfied: requests in c:\_metric_pler+\pler_env\lib\site-packages (from dash>=2.6.0->open3d>=0.15.1->-r requirements.txt (line 2)) (2.32.5)
Requirement already satisfied: retrying in c:\_metric_pler+\pler_env\lib\site-packages (from dash>=2.6.0->open3d>=0.15.1->-r requirements.txt (line 2)) (1.4.2)
Requirement already satisfied: nest-asyncio in c:\_metric_pler+\pler_env\lib\site-packages (from dash>=2.6.0->open3d>=0.15.1->-r requirements.txt (line 2)) (1.6.0)
Requirement already satisfied: setuptools in c:\_metric_pler+\pler_env\lib\site-packages (from dash>=2.6.0->open3d>=0.15.1->-r requirements.txt (line 2)) (65.5.0)
Requirement already satisfied: blinker>=1.9.0 in c:\_metric_pler+\pler_env\lib\site-packages (from flask>=3.0.0->open3d>=0.15.1->-r requirements.txt (line 2)) (1.9.0)
Requirement already satisfied: click>=8.1.3 in c:\_metric_pler+\pler_env\lib\site-packages (from flask>=3.0.0->open3d>=0.15.1->-r requirements.txt (line 2)) (8.3.1)
Requirement already satisfied: itsdangerous>=2.2.0 in c:\_metric_pler+\pler_env\lib\site-packages (from flask>=3.0.0->open3d>=0.15.1->-r requirements.txt (line 2)) (2.2.0)
Requirement already satisfied: jinja2>=3.1.2 in c:\_metric_pler+\pler_env\lib\site-packages (from flask>=3.0.0->open3d>=0.15.1->-r requirements.txt (line 2)) (3.1.6)
Requirement already satisfied: markupsafe>=2.1.1 in c:\_metric_pler+\pler_env\lib\site-packages (from flask>=3.0.0->open3d>=0.15.1->-r requirements.txt (line 2)) (3.0.3)
Requirement already satisfied: comm>=0.1.3 in c:\_metric_pler+\pler_env\lib\site-packages (from ipywidgets>=8.0.4->open3d>=0.15.1->-r requirements.txt (line 2)) (0.2.3)
Requirement already satisfied: ipython>=6.1.0 in c:\_metric_pler+\pler_env\lib\site-packages (from ipywidgets>=8.0.4->open3d>=0.15.1->-r requirements.txt (line 2)) (8.37.0)
Requirement already satisfied: traitlets>=4.3.1 in c:\_metric_pler+\pler_env\lib\site-packages (from ipywidgets>=8.0.4->open3d>=0.15.1->-r requirements.txt (line 2)) (5.14.3)
Requirement already satisfied: widgetsnbextension~=4.0.14 in c:\_metric_pler+\pler_env\lib\site-packages (from ipywidgets>=8.0.4->open3d>=0.15.1->-r requirements.txt (line 2)) (4.0.15)  
Requirement already satisfied: jupyterlab_widgets~=3.0.15 in c:\_metric_pler+\pler_env\lib\site-packages (from ipywidgets>=8.0.4->open3d>=0.15.1->-r requirements.txt (line 2)) (3.0.16)  
Requirement already satisfied: decorator in c:\_metric_pler+\pler_env\lib\site-packages (from ipython>=6.1.0->ipywidgets>=8.0.4->open3d>=0.15.1->-r requirements.txt (line 2)) (5.2.1)
Requirement already satisfied: jedi>=0.16 in c:\_metric_pler+\pler_env\lib\site-packages (from ipython>=6.1.0->ipywidgets>=8.0.4->open3d>=0.15.1->-r requirements.txt (line 2)) (0.19.2)
Requirement already satisfied: matplotlib-inline in c:\_metric_pler+\pler_env\lib\site-packages (from ipython>=6.1.0->ipywidgets>=8.0.4->open3d>=0.15.1->-r requirements.txt (line 2)) (0.2.1)
Requirement already satisfied: prompt_toolkit<3.1.0,>=3.0.41 in c:\_metric_pler+\pler_env\lib\site-packages (from ipython>=6.1.0->ipywidgets>=8.0.4->open3d>=0.15.1->-r requirements.txt (line 2)) (3.0.52)
Requirement already satisfied: stack_data in c:\_metric_pler+\pler_env\lib\site-packages (from ipython>=6.1.0->ipywidgets>=8.0.4->open3d>=0.15.1->-r requirements.txt (line 2)) (0.6.3)
Requirement already satisfied: wcwidth in c:\_metric_pler+\pler_env\lib\site-packages (from prompt_toolkit<3.1.0,>=3.0.41->ipython>=6.1.0->ipywidgets>=8.0.4->open3d>=0.15.1->-r requirements.txt (line 2)) (0.2.14)
Requirement already satisfied: parso<0.9.0,>=0.8.4 in c:\_metric_pler+\pler_env\lib\site-packages (from jedi>=0.16->ipython>=6.1.0->ipywidgets>=8.0.4->open3d>=0.15.1->-r requirements.txt (line 2)) (0.8.5)
Requirement already satisfied: fastjsonschema in c:\_metric_pler+\pler_env\lib\site-packages (from nbformat>=5.7.0->open3d>=0.15.1->-r requirements.txt (line 2)) (2.21.2)
Requirement already satisfied: jsonschema>=2.6 in c:\_metric_pler+\pler_env\lib\site-packages (from nbformat>=5.7.0->open3d>=0.15.1->-r requirements.txt (line 2)) (4.25.1)
Requirement already satisfied: jupyter-core in c:\_metric_pler+\pler_env\lib\site-packages (from nbformat>=5.7.0->open3d>=0.15.1->-r requirements.txt (line 2)) (5.9.1)
Requirement already satisfied: attrs>=22.2.0 in c:\_metric_pler+\pler_env\lib\site-packages (from jsonschema>=2.6->nbformat>=5.7.0->open3d>=0.15.1->-r requirements.txt (line 2)) (25.4.0)  
Requirement already satisfied: jsonschema-specifications>=2023.03.6 in c:\_metric_pler+\pler_env\lib\site-packages (from jsonschema>=2.6->nbformat>=5.7.0->open3d>=0.15.1->-r requirements.txt (line 2)) (2025.9.1)
Requirement already satisfied: referencing>=0.28.4 in c:\_metric_pler+\pler_env\lib\site-packages (from jsonschema>=2.6->nbformat>=5.7.0->open3d>=0.15.1->-r requirements.txt (line 2)) (0.37.0)
Requirement already satisfied: rpds-py>=0.7.1 in c:\_metric_pler+\pler_env\lib\site-packages (from jsonschema>=2.6->nbformat>=5.7.0->open3d>=0.15.1->-r requirements.txt (line 2)) (0.29.0)  
Requirement already satisfied: narwhals>=1.15.1 in c:\_metric_pler+\pler_env\lib\site-packages (from plotly>=5.0.0->dash>=2.6.0->open3d>=0.15.1->-r requirements.txt (line 2)) (2.12.0)
Requirement already satisfied: six>=1.5 in c:\_metric_pler+\pler_env\lib\site-packages (from python-dateutil>=2.7->matplotlib>=3.4.0->-r requirements.txt (line 5)) (1.17.0)
Requirement already satisfied: zipp>=3.20 in c:\_metric_pler+\pler_env\lib\site-packages (from importlib-metadata->dash>=2.6.0->open3d>=0.15.1->-r requirements.txt (line 2)) (3.23.0)
Requirement already satisfied: platformdirs>=2.5 in c:\_metric_pler+\pler_env\lib\site-packages (from jupyter-core->nbformat>=5.7.0->open3d>=0.15.1->-r requirements.txt (line 2)) (4.5.0)
Requirement already satisfied: charset_normalizer<4,>=2 in c:\_metric_pler+\pler_env\lib\site-packages (from requests->dash>=2.6.0->open3d>=0.15.1->-r requirements.txt (line 2)) (3.4.4)
Requirement already satisfied: idna<4,>=2.5 in c:\_metric_pler+\pler_env\lib\site-packages (from requests->dash>=2.6.0->open3d>=0.15.1->-r requirements.txt (line 2)) (3.11)
Requirement already satisfied: urllib3<3,>=1.21.1 in c:\_metric_pler+\pler_env\lib\site-packages (from requests->dash>=2.6.0->open3d>=0.15.1->-r requirements.txt (line 2)) (2.5.0)
Requirement already satisfied: certifi>=2017.4.17 in c:\_metric_pler+\pler_env\lib\site-packages (from requests->dash>=2.6.0->open3d>=0.15.1->-r requirements.txt (line 2)) (2025.11.12)  
Requirement already satisfied: executing>=1.2.0 in c:\_metric_pler+\pler_env\lib\site-packages (from stack_data->ipython>=6.1.0->ipywidgets>=8.0.4->open3d>=0.15.1->-r requirements.txt (line 2)) (2.2.1)
Requirement already satisfied: asttokens>=2.1.0 in c:\_metric_pler+\pler_env\lib\site-packages (from stack_data->ipython>=6.1.0->ipywidgets>=8.0.4->open3d>=0.15.1->-r requirements.txt (line 2)) (3.0.1)
Requirement already satisfied: pure-eval in c:\_metric_pler+\pler_env\lib\site-packages (from stack_data->ipython>=6.1.0->ipywidgets>=8.0.4->open3d>=0.15.1->-r requirements.txt (line 2)) (0.2.3)
(pler_env) PS C:\_Metric_PLER+> python sphere_analysis.py --num-models 15 --reference models/Sphere.obj --pattern "models/Sphere*{}.obj"
🔍 Анализ 15 моделей...
✅ Эталон загружен: 763192 вершин
📊 Модель 1/15: Sphere_1.obj
❌ Ошибка: 'dict' object has no attribute 'norm'
📊 Модель 2/15: Sphere_2.obj
❌ Ошибка: 'dict' object has no attribute 'norm'
📊 Модель 3/15: Sphere_3.obj
❌ Ошибка: 'dict' object has no attribute 'norm'
📊 Модель 4/15: Sphere_4.obj
❌ Ошибка: 'dict' object has no attribute 'norm'
📊 Модель 5/15: Sphere_5.obj
❌ Ошибка: 'dict' object has no attribute 'norm'
📊 Модель 6/15: Sphere_6.obj
❌ Ошибка: 'dict' object has no attribute 'norm'
📊 Модель 7/15: Sphere_7.obj
❌ Ошибка: 'dict' object has no attribute 'norm'
📊 Модель 8/15: Sphere_8.obj
❌ Ошибка: 'dict' object has no attribute 'norm'
📊 Модель 9/15: Sphere_9.obj
❌ Ошибка: 'dict' object has no attribute 'norm'
📊 Модель 10/15: Sphere_10.obj
❌ Ошибка: 'dict' object has no attribute 'norm'
📊 Модель 11/15: Sphere_11.obj
❌ Ошибка: 'dict' object has no attribute 'norm'
📊 Модель 12/15: Sphere_12.obj
❌ Ошибка: 'dict' object has no attribute 'norm'
📊 Модель 13/15: Sphere_13.obj
❌ Ошибка: 'dict' object has no attribute 'norm'
📊 Модель 13/15: Sphere_13.obj
📊 Модель 13/15: Sphere_13.obj
❌ Ошибка: 'dict' object has no attribute 'norm'
📊 Модель 14/15: Sphere_14.obj
❌ Ошибка: 'dict' object has no attribute 'norm'
📊 Модель 15/15: Sphere_15.obj
📊 Модель 15/15: Sphere_15.obj
❌ Ошибка: 'dict' object has no attribute 'norm'
💾 Результаты сохранены в sphere_analysis_results.csv

📊 СВОДНАЯ СТАТИСТИКА:
Моделей проанализировано: 15
MCM_AAD: 0.000374 ± 0.000981
MCM_SSD: 0.000202 ± 0.000529
Ошибка объема: 7.01% ± 18.15%
Корреляция вершин-MCM_AAD: 0.222
📈 Графики трендов сохранены в sphere_analysis_trends.png

✅ Анализ завершен!
(pler_env) PS C:\_Metric_PLER+> python test_improvements.py
🧪 Тестирование улучшений PLER...
2025-12-04 11:46:03,114 - INFO - 🧹 Очищено 5 просроченных кэшей
2025-12-04 11:46:03,116 - WARNING - ❌ CUDA и OpenCL недоступны, используем CPU
2025-12-04 11:46:03,116 - INFO - ✅ Инициализирован PLERMetric (потоков: 7, GPU: False)

📊 ИНФОРМАЦИЯ О СИСТЕМЕ:
cpu_cores: 8
worker_threads: 7
gpu_available: False
cache_enabled: True
supported_formats: ['.ply', '.gltf', '.obj', '.3ds', '.dae', '.fbx', '.stl', '.off', '.glb']
available: False
total_files: 0
total_size_mb: 0.0
oldest_file: None
newest_file: None
cache_dir: .pler_cache

🔍 ТЕСТ 1: simple_reference.obj vs simple_distorted.obj
2025-12-04 11:46:03,119 - INFO - 🔍 Загрузка и нормализация моделей...
2025-12-04 11:46:03,140 - INFO - ✅ Успешно загружено: simple_reference.obj - 114 вершин, 224 граней
2025-12-04 11:46:03,154 - INFO - ✅ Успешно загружено: simple_distorted.obj - 114 вершин, 224 граней
2025-12-04 11:46:03,154 - INFO - 📊 Загружено: ref=114 вершин, dist=114 вершин
2025-12-04 11:46:03,154 - INFO - 🎯 Используется 34 лучей
2025-12-04 11:46:03,155 - INFO - 📏 Выполнение параллельного ray casting...
Первый запуск: 12.79 dB, 0.05с, GPU: False, Кэш: False
2025-12-04 11:46:03,183 - INFO - ✅ Используем закэшированный результат (ключ: 13ed2e75...)
Второй запуск: 12.79 dB, 0.01с, GPU: False, Кэш: True
Ускорение: 3.7x

2025-12-04 11:46:03,200 - INFO - ✅ Успешно загружено: heavily_distorted.obj - 114 вершин, 224 граней
2025-12-04 11:46:03,200 - INFO - 📊 Загружено: ref=114 вершин, dist=114 вершин  
2025-12-04 11:46:03,200 - INFO - 🎯 Используется 34 лучей
2025-12-04 11:46:03,200 - INFO - 📏 Выполнение параллельного ray casting...
Первый запуск: 5.74 dB, 0.02с, GPU: False, Кэш: False
2025-12-04 11:46:03,218 - INFO - ✅ Используем закэшированный результат (ключ: ebf08136...)
Второй запуск: 5.74 dB, 0.01с, GPU: False, Кэш: True
Ускорение: 1.8x
2025-12-04 11:46:03,219 - INFO - ✅ Кэш загрузчика очищен
2025-12-04 11:46:03,220 - INFO - 🧹 Очищено 2 файлов кэша
2025-12-04 11:46:03,220 - INFO - ✅ Все кэши очищены

✅ Тестирование завершено
(pler_env) PS C:\_Metric_PLER+>

pler*env\Scripts\activate  
python research_mode.py --num-models 9 --reference models/Cone.obj --pattern "models/Cone*{}.obj"
