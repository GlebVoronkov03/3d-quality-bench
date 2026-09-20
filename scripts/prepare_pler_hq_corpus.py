"""
Каркас корпуса PLER-HQ: каталог на 50 эталонов, 5 примитивов, структура папок.
Не копирует дубликаты из data/high_quality. peach.fbx не конвертирует.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "DataBase"
HQ = ROOT / "data" / "high_quality"
META = DB / "metadata"
REF = DB / "references"
DIST = DB / "distorted"
TEX = DB / "textures"

CATEGORIES = {
    "sampling": "Контроль дискретизации",
    "canonical": "Канонические эталоны",
    "acquisition": "Артефакты оцифровки",
    "thin_feature": "Тонкие структуры",
    "sharp_feature": "Жёсткие рёбра",
    "perceptual": "Перцептивное маскирование",
}

# Жёсткие квоты (сумма = 50). Каталог catalog.csv — источник истины; этот скрипт его не перезаписывает.
QUOTAS = {
    "sampling": 6,
    "canonical": 9,
    "acquisition": 8,
    "thin_feature": 6,
    "sharp_feature": 8,
    "perceptual": 13,
}

# 10 градаций полного корпуса
LADDER = {
    "Decimation": [0.80, 0.50, 0.35, 0.25, 0.15, 0.10, 0.05, 0.025, 0.01, 0.005],
    "Noise": [0.0005, 0.001, 0.002, 0.004, 0.007, 0.010, 0.015, 0.022, 0.032, 0.050],
    "Smoothing": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Combined_decimation": [0.80, 0.50, 0.35, 0.25, 0.15, 0.10, 0.05, 0.025, 0.01, 0.005],
    "Combined_noise": [0.0005, 0.001, 0.002, 0.004, 0.007, 0.010, 0.015, 0.022, 0.032, 0.050],
}


def _row(
    slot: int,
    slug: str,
    name_ru: str,
    category: str,
    source: str,
    license_id: str,
    rel_path: str,
    status: str,
    notes: str = "",
) -> dict:
    return {
        "slot": slot,
        "slug": slug,
        "name_ru": name_ru,
        "category": category,
        "source": source,
        "license": license_id,
        "path": rel_path,
        "status": status,
        "notes": notes,
    }


def build_catalog() -> list[dict]:
    """50 слотов: текущие файлы + примитивы + публичные + ожидание пользователя."""
    rows: list[dict] = []

    # --- already on disk in DataBase / high_quality ---
    existing = [
        _row(1, "uv-sphere-dense", "UV-сфера (высокая плотность)", "primitives",
             "проект QV / авторская", "project", "DataBase/Sphere_523276_vertices.obj", "present"),
        _row(2, "stanford-bunny", "Стенфордский заяц", "academic_cg",
             "Stanford 3D Scanning Repository", "stanford-research", "DataBase/stanford-bunny.obj", "present"),
        _row(3, "teapot", "Чайник Utah / 3ds Max", "academic_cg",
             "Utah teapot (Newell), вариант 3ds Max", "fair-use-research", "DataBase/teapot.obj", "present"),
        _row(4, "suzanne", "Suzanne (Blender)", "academic_cg",
             "Blender Foundation", "GPL-CC-compatible", "data/high_quality/suzanne.obj", "present",
             "Файл не копируется из high_quality (запрет на дубли)"),
        _row(5, "bust", "Бюст (ГМИИ им. Пушкина, фотограмметрия)", "heritage_scan",
             "авторская фотограмметрия", "author-all-rights", "DataBase/bust.obj", "present"),
        _row(6, "relief", "Рельеф (цветочный, фотограмметрия)", "heritage_scan",
             "авторская фотограмметрия", "author-all-rights", "DataBase/Рельеф.obj", "present"),
        _row(7, "sarcophagus", "Греческий саркофаг", "heritage_scan",
             "авторская фотограмметрия", "author-all-rights", "DataBase/greek sarcophagus.obj", "present"),
        _row(8, "egorov-lidar", "LiDAR Егорова", "heritage_scan",
             "авторский LiDAR", "author-all-rights", "DataBase/Egorov_LiDAR_1/textured_output.obj", "present"),
        _row(9, "automaton", "Low-poly автоматон", "human_character",
             "авторский / ассет проекта", "author-or-cc", "DataBase/low-poly automaton.obj", "present"),
        _row(10, "doll", "Кукла", "human_character",
             "авторский / ассет проекта", "author-or-cc", "DataBase/Кукла.obj", "present"),
        _row(11, "skeleton", "Скелет", "human_character",
             "авторский / ассет проекта", "author-or-cc", "DataBase/скелет.obj", "present"),
        _row(12, "moncey", "Moncey (скульптура)", "heritage_scan",
             "авторский / ассет проекта", "author-or-cc", "DataBase/moncey.obj", "present"),
        _row(13, "murano-chair", "Кресло Murano", "everyday",
             "авторский / ассет проекта", "author-or-cc", "DataBase/Murano chair.obj", "present"),
        _row(14, "bench", "Скамейка", "everyday",
             "авторский / ассет проекта", "author-or-cc", "DataBase/скамейка.obj", "present"),
        _row(15, "well", "Каменный колодец", "everyday",
             "авторский / ассет проекта", "author-or-cc", "DataBase/каменный колодец.obj", "present"),
        _row(16, "boots", "Сапоги", "everyday",
             "авторский / ассет проекта", "author-or-cc", "DataBase/bootse.obj", "present"),
        _row(17, "belt", "Ремень", "everyday",
             "авторский / ассет проекта", "author-or-cc", "DataBase/belt.obj", "present"),
        _row(18, "axe", "Топор", "everyday",
             "авторский / ассет проекта", "author-or-cc", "DataBase/axe.obj", "present"),
        _row(19, "bull-skull", "Череп быка", "organic",
             "авторский / ассет проекта", "author-or-cc", "DataBase/череп быка.obj", "present"),
        _row(20, "peach", "Персик", "organic",
             "авторский / ассет проекта", "author-or-cc", "DataBase/peach.fbx", "pending_conversion",
             "Пользователь конвертирует FBX→OBJ"),
    ]

    # 5 примитивов, которые генерирует этот скрипт
    generated = [
        _row(21, "cube", "Куб", "primitives", "синтез trimesh", "CC0-generated",
             "DataBase/references/primitives/cube.obj", "generated"),
        _row(22, "icosphere", "ICO-сфера", "primitives", "синтез trimesh", "CC0-generated",
             "DataBase/references/primitives/icosphere.obj", "generated"),
        _row(23, "torus", "Тор (пончик)", "primitives", "синтез trimesh", "CC0-generated",
             "DataBase/references/primitives/torus.obj", "generated"),
        _row(24, "cylinder", "Цилиндр", "primitives", "синтез trimesh", "CC0-generated",
             "DataBase/references/primitives/cylinder.obj", "generated"),
        _row(25, "cone", "Конус", "primitives", "синтез trimesh", "CC0-generated",
             "DataBase/references/primitives/cone.obj", "generated"),
    ]

    # 24 публичные модели с явной лицензией (скачивание — download_public_models.py)
    public = [
        _row(26, "armadillo", "Armadillo", "academic_cg",
             "Stanford 3D Scanning Repository / Jacobson mirror", "stanford-research",
             "DataBase/references/academic_cg/armadillo.obj", "pending_download"),
        _row(27, "happy-buddha", "Happy Buddha", "academic_cg",
             "Stanford 3D Scanning Repository / Jacobson mirror", "stanford-research",
             "DataBase/references/academic_cg/happy.obj", "pending_download"),
        _row(28, "xyzrgb-dragon", "XYZRGB Dragon", "academic_cg",
             "Stanford 3D Scanning Repository / Jacobson mirror", "stanford-research",
             "DataBase/references/academic_cg/xyzrgb_dragon.obj", "pending_download"),
        _row(29, "lucy", "Lucy (decimated ≤100k)", "academic_cg",
             "Stanford 3D Scanning Repository / Jacobson mirror", "stanford-research",
             "DataBase/references/academic_cg/lucy.obj", "pending_download"),
        _row(30, "max-planck", "Max Planck", "academic_cg",
             "AIM@SHAPE / Jacobson mirror", "aim-at-shape-research",
             "DataBase/references/academic_cg/max-planck.obj", "pending_download"),
        _row(31, "bimba", "Bimba", "heritage_scan",
             "AIM@SHAPE / Jacobson mirror", "aim-at-shape-research",
             "DataBase/references/heritage_scan/bimba.obj", "pending_download"),
        _row(32, "igea", "Igea", "heritage_scan",
             "Cyberware / Jacobson mirror", "research-attribution",
             "DataBase/references/heritage_scan/igea.obj", "pending_download"),
        _row(33, "nefertiti", "Nefertiti", "heritage_scan",
             "скан / Jacobson mirror", "research-attribution",
             "DataBase/references/heritage_scan/nefertiti.obj", "pending_download"),
        _row(34, "ogre", "Ogre", "human_character",
             "Jacobson common-3d-test-models", "research-attribution",
             "DataBase/references/human_character/ogre.obj", "pending_download"),
        _row(35, "homer", "Homer", "human_character",
             "Jacobson common-3d-test-models", "research-attribution",
             "DataBase/references/human_character/homer.obj", "pending_download"),
        _row(36, "cheburashka", "Чебурашка", "human_character",
             "Jacobson common-3d-test-models", "research-attribution",
             "DataBase/references/human_character/cheburashka.obj", "pending_download"),
        _row(37, "woody", "Woody", "human_character",
             "Jacobson common-3d-test-models", "research-attribution",
             "DataBase/references/human_character/woody.obj", "pending_download"),
        _row(38, "horse", "Лошадь", "organic",
             "Jacobson common-3d-test-models", "research-attribution",
             "DataBase/references/organic/horse.obj", "pending_download"),
        _row(39, "spot", "Spot (корова Crane)", "organic",
             "Keenan Crane / Jacobson mirror", "cc-by-or-research",
             "DataBase/references/organic/spot.obj", "pending_download"),
        _row(40, "alligator", "Аллигатор", "organic",
             "Jacobson common-3d-test-models", "research-attribution",
             "DataBase/references/organic/alligator.obj", "pending_download"),
        _row(41, "beast", "Beast", "organic",
             "Jacobson common-3d-test-models", "research-attribution",
             "DataBase/references/organic/beast.obj", "pending_download"),
        _row(42, "fandisk", "Fandisk", "mechanical",
             "классический CAD-тест / Jacobson", "research-attribution",
             "DataBase/references/mechanical/fandisk.obj", "pending_download"),
        _row(43, "rocker-arm", "Rocker arm", "mechanical",
             "AIM@SHAPE / Jacobson mirror", "aim-at-shape-research",
             "DataBase/references/mechanical/rocker-arm.obj", "pending_download"),
        _row(44, "beetle", "Beetle (автомобиль)", "mechanical",
             "Jacobson common-3d-test-models", "research-attribution",
             "DataBase/references/mechanical/beetle.obj", "pending_download"),
        _row(45, "stanford-dragon", "Stanford Dragon", "academic_cg",
             "Stanford 3D Scanning Repository", "stanford-research",
             "DataBase/references/academic_cg/stanford-dragon.obj", "pending_download"),
        _row(46, "fertility", "Fertility", "heritage_scan",
             "AIM@SHAPE / Visual Computing Lab ISTI-CNR", "aim-at-shape-research",
             "DataBase/references/heritage_scan/fertility.obj", "pending_download"),
        _row(47, "aio-printer-test", "All In One 3D printer test (majda107 original 3rd gen)", "mechanical",
             "majda107 / Thingiverse 2656594 / GitHub majda107/3d-printer-test", "cc-by",
             "DataBase/references/mechanical/aio_printer_test.obj", "pending_download"),
        _row(48, "thingi-b", "Sam's Gears (Thingi10K 90275)", "mechanical",
             "Thingi10K file_id=90275 thing_id=30981", "cc-by",
             "DataBase/references/mechanical/thingi_b.obj", "pending_download"),
        _row(49, "male-anatomy", "Male anatomy figure (C.J. Goldman, A-pose)", "human_character",
             "Sketchfab e39449c8c59346788834b94706faf4bb", "cc-by",
             "DataBase/references/human_character/male-anatomy.obj", "pending_download"),
        _row(50, "thingi-a", "Planetary gear ring (Thingi10K 59228)", "mechanical",
             "Thingi10K file_id=59228 thing_id=18291", "cc-by",
             "DataBase/references/mechanical/thingi_a.obj", "pending_download",
             "Замена слота дополнительной фотограмметрии"),
    ]

    rows.extend(existing)
    rows.extend(generated)
    rows.extend(public)
    assert len(rows) == 50, len(rows)
    assert len({r["slot"] for r in rows}) == 50
    return rows


def generate_primitives() -> None:
    import trimesh

    out = REF / "primitives"
    out.mkdir(parents=True, exist_ok=True)

    meshes = {
        "cube": trimesh.creation.box(extents=[1.0, 1.0, 1.0]),
        "icosphere": trimesh.creation.icosphere(subdivisions=4, radius=0.5),
        "torus": trimesh.creation.torus(major_radius=0.45, minor_radius=0.18, major_sections=64, minor_sections=32),
        "cylinder": trimesh.creation.cylinder(radius=0.35, height=1.0, sections=64),
        "cone": trimesh.creation.cone(radius=0.45, height=1.0, sections=64),
    }
    for name, mesh in meshes.items():
        mesh.vertices = mesh.vertices - mesh.centroid
        path = out / f"{name}.obj"
        mesh.export(path)
        print(f"wrote {path} V={len(mesh.vertices)} F={len(mesh.faces)}")


def write_metadata(rows: list[dict]) -> None:
    META.mkdir(parents=True, exist_ok=True)
    cat = META / "catalog.csv"
    with cat.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (META / "distortion_ladder.json").write_text(
        json.dumps(LADDER, indent=2), encoding="utf-8"
    )
    (META / "quotas.json").write_text(
        json.dumps({"categories": CATEGORIES, "quotas": QUOTAS}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"catalog -> {cat}")


def ensure_dirs() -> None:
    for key in CATEGORIES:
        (REF / key).mkdir(parents=True, exist_ok=True)
    DIST.mkdir(parents=True, exist_ok=True)
    TEX.mkdir(parents=True, exist_ok=True)
    META.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ensure_dirs()
    generate_primitives()
    cat = META / "catalog.csv"
    if cat.exists():
        print(f"catalog.csv already exists ({cat}) — not overwritten")
        (META / "distortion_ladder.json").write_text(json.dumps(LADDER, indent=2), encoding="utf-8")
        qpath = META / "quotas.json"
        if not qpath.exists():
            qpath.write_text(
                json.dumps({"categories": CATEGORIES, "quotas": QUOTAS}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    else:
        write_metadata(build_catalog())
    print("PLER-HQ corpus skeleton ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
