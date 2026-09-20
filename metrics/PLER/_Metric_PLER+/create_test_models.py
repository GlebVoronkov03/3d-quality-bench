# create_test_models.py
import numpy as np
import os
from pathlib import Path

def create_simple_sphere(radius=1.0, sectors=16, stacks=8):
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
            second_next = (i + 1) * sectors + next_j
            
            # Два треугольника на каждую ячейку
            if i != 0:
                faces.append([first, second, first_next])
            if i != stacks - 1:
                faces.append([first_next, second, second_next])
    
    return vertices, faces

def save_obj(filename, vertices, faces):
    """Сохранение в OBJ формат"""
    # Создаем папку если не существует
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# Simple test mesh\n")
        f.write(f"# Vertices: {len(vertices)}, Faces: {len(faces)}\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def create_test_models():
    """Создание тестовых моделей"""
    print("🎯 Создание тестовых моделей...")
    
    # Создание сферы
    vertices, faces = create_simple_sphere()
    
    # Эталонная модель
    save_obj("models/simple_reference.obj", vertices, faces)
    print(f"✅ Эталонная модель: {len(vertices)} вершин, {len(faces)} граней")
    
    # Искаженная модель (немного сжатая по разным осям)
    distorted_vertices = [[v[0]*0.8, v[1]*1.2, v[2]*0.9] for v in vertices]
    save_obj("models/simple_distorted.obj", distorted_vertices, faces)
    print(f"✅ Искаженная модель: {len(distorted_vertices)} вершин, {len(faces)} граней")
    
    # Сильно искаженная модель для тестирования
    heavily_distorted = [[v[0]*0.5, v[1]*1.5, v[2]*0.7] for v in vertices]
    save_obj("models/heavily_distorted.obj", heavily_distorted, faces)
    print(f"✅ Сильно искаженная модель: {len(heavily_distorted)} вершин, {len(faces)} граней")
    
    print("\n📁 Созданные модели:")
    print("  • models/simple_reference.obj")
    print("  • models/simple_distorted.obj") 
    print("  • models/heavily_distorted.obj")
    print("\n🎯 Для тестирования выполните:")
    print("  python pler_metric.py models/simple_reference.obj models/simple_distorted.obj")

if __name__ == "__main__":
    create_test_models()