"""
Blender 4.2 offscreen clay-render для выбранных эталонов.
Запуск:
  blender -b -P scripts/blender_render_refs.py -- slug1 slug2 ...
Без аргументов: relief, egorov-lidar, woody, alligator, well, automaton.
Файлы >120 МБ: если есть docs/pptx_assets/_preview/{slug}.obj — берём его.
"""
from __future__ import annotations

import csv
import math
import shutil
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

ROOT = Path(r"C:\_PLER-2.0_experiment")
CAT = ROOT / "DataBase" / "metadata" / "catalog.csv"
OUT = ROOT / "docs" / "pptx_assets" / "refs"
PREV = ROOT / "docs" / "pptx_assets" / "_preview"
BG = (0.957, 0.945, 0.918, 1.0)
CLAY = (0.46, 0.48, 0.51, 1.0)

DEFAULT = ["relief", "egorov-lidar", "woody", "alligator", "well", "automaton"]


def ascii_copy(src: Path) -> Path:
    try:
        src.resolve().as_posix().encode("ascii")
        return src
    except UnicodeEncodeError:
        tmp = ROOT / "docs" / "pptx_assets" / "_tmp_ascii"
        tmp.mkdir(parents=True, exist_ok=True)
        dst = tmp / f"b_{abs(hash(src.as_posix())) % 10**8}.obj"
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
        return dst


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_obj(path: Path):
    src = ascii_copy(path)
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=str(src))
    else:
        bpy.ops.import_scene.obj(filepath=str(src))


def combined_objects():
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("no mesh")
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def clay_mat():
    mat = bpy.data.materials.new("clay")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = CLAY
    bsdf.inputs["Roughness"].default_value = 0.58
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.12
    elif "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = 0.12
    return mat


def sampled_world_coords(obj, cap: int = 180_000) -> np.ndarray:
    mesh = obj.data
    n = len(mesh.vertices)
    if n == 0:
        raise RuntimeError("empty verts")
    step = max(1, n // cap)
    mw = np.array(obj.matrix_world, dtype=np.float64)
    coords = np.empty(((n + step - 1) // step, 3), dtype=np.float64)
    j = 0
    for i in range(0, n, step):
        co = mesh.vertices[i].co
        x, y, z, w = mw @ np.array((co.x, co.y, co.z, 1.0))
        coords[j] = (x / w, y / w, z / w) if w else (x, y, z)
        j += 1
    return coords[:j]


def bake_and_center(obj) -> tuple[np.ndarray, np.ndarray]:
    bpy.context.view_layer.update()
    coords = sampled_world_coords(obj)
    lo = np.percentile(coords, 0.6, axis=0)
    hi = np.percentile(coords, 99.4, axis=0)
    center = 0.5 * (lo + hi)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    for v in obj.data.vertices:
        v.co.x -= float(center[0])
        v.co.y -= float(center[1])
        v.co.z -= float(center[2])
    obj.data.update()
    lo = lo - center
    hi = hi - center
    ext = hi - lo
    if float(np.min(ext)) / max(float(np.max(ext)), 1e-12) < 0.16:
        pts = sampled_world_coords(obj)
        c = pts.mean(axis=0)
        x = pts - c
        _, _, vh = np.linalg.svd(x, full_matrices=False)
        thin = vh[-1]
        target = np.array([0.0, 0.0, 1.0])
        axis = np.cross(thin, target)
        nrm = np.linalg.norm(axis)
        if nrm > 1e-8:
            axis = axis / nrm
            ang = math.acos(float(np.clip(np.dot(thin, target), -1.0, 1.0)))
            rot = Matrix.Rotation(ang, 4, Vector(axis.tolist()))
            obj.matrix_world = rot @ obj.matrix_world
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            coords = sampled_world_coords(obj)
            lo = np.percentile(coords, 0.6, axis=0)
            hi = np.percentile(coords, 99.4, axis=0)
            center2 = 0.5 * (lo + hi)
            for v in obj.data.vertices:
                v.co.x -= float(center2[0])
                v.co.y -= float(center2[1])
                v.co.z -= float(center2[2])
            obj.data.update()
            lo = lo - center2
            hi = hi - center2
    return lo, hi


def fit_camera(obj, cam, lo, hi):
    ext = hi - lo
    span = float(np.max(ext)) or 1.0
    radius = 0.5 * float(np.linalg.norm(ext)) or span
    dist = max(span, radius) * 2.4
    thin = float(np.min(ext)) / max(span, 1e-12)
    if thin < 0.16:
        loc = Vector((dist * 0.12, -dist * 0.42, dist * 1.15))
        scale = span * 1.18
    else:
        loc = Vector((dist * 0.62, -dist * 0.82, dist * 0.48))
        scale = span * 1.28
    cam.location = loc
    direction = Vector((0.0, 0.0, 0.0)) - loc
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = scale
    cam.data.clip_start = 0.001
    cam.data.clip_end = dist * 40.0


def setup_scene(scene, dst: Path):
    engines = ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE")
    for eng in engines:
        try:
            scene.render.engine = eng
            break
        except Exception:
            continue
    scene.render.resolution_x = 720
    scene.render.resolution_y = 720
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.filepath = str(dst)
    scene.display_settings.display_device = "sRGB"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    if hasattr(scene, "eevee"):
        if hasattr(scene.eevee, "taa_render_samples"):
            scene.eevee.taa_render_samples = 8
        if hasattr(scene.eevee, "use_gtao"):
            scene.eevee.use_gtao = True
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    if hasattr(scene.display.shading, "background_type"):
        scene.display.shading.background_type = "VIEWPORT"
        scene.display.shading.background_color = BG[:3]
    world = bpy.data.worlds.new("w")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = BG
        bg.inputs[1].default_value = 1.15
    world.color = BG[:3]


def add_lights(scene, cam):
    key = bpy.data.lights.new("key", "AREA")
    key.energy = 420.0
    key.size = 6.0
    key_o = bpy.data.objects.new("key", key)
    scene.collection.objects.link(key_o)
    key_o.location = cam.location + Vector((2.0, -1.2, 3.5))
    fill = bpy.data.lights.new("fill", "AREA")
    fill.energy = 140.0
    fill.size = 8.0
    fill_o = bpy.data.objects.new("fill", fill)
    scene.collection.objects.link(fill_o)
    fill_o.location = Vector((-4.0, 3.5, 2.0))
    rim = bpy.data.lights.new("rim", "AREA")
    rim.energy = 90.0
    rim.size = 5.0
    rim_o = bpy.data.objects.new("rim", rim)
    scene.collection.objects.link(rim_o)
    rim_o.location = Vector((0.5, 4.0, 5.0))


def render_one(src: Path, dst: Path):
    print("BLENDER", src.name, flush=True)
    reset()
    scene = bpy.context.scene
    setup_scene(scene, dst)
    import_obj(src)
    obj = combined_objects()
    mat = clay_mat()
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    if len(obj.data.polygons) > 40_000:
        bpy.ops.object.shade_smooth()
    lo, hi = bake_and_center(obj)
    print(f"  aabb {lo} .. {hi}", flush=True)
    cam_data = bpy.data.cameras.new("cam")
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    fit_camera(obj, cam, lo, hi)
    add_lights(scene, cam)
    bpy.ops.render.render(write_still=True)
    print("  wrote", dst, flush=True)


def resolve_src(rec: dict) -> Path:
    slug = rec["slug"]
    preview = PREV / f"{slug}.obj"
    src = ROOT / rec["path"]
    if preview.exists() and preview.stat().st_size > 1000:
        print("  using preview", preview, flush=True)
        return preview
    return src


def argv_slugs():
    if "--" in sys.argv:
        extra = [a for a in sys.argv[sys.argv.index("--") + 1 :] if a]
        return extra or DEFAULT
    return DEFAULT


def main():
    slugs = argv_slugs()
    rows = {r["slug"]: r for r in csv.DictReader(CAT.open(encoding="utf-8-sig"))}
    OUT.mkdir(parents=True, exist_ok=True)
    for slug in slugs:
        rec = rows[slug]
        src = resolve_src(rec)
        dst = OUT / f"{slug}.png"
        if not src.exists():
            print("missing", src)
            continue
        try:
            render_one(src, dst)
        except Exception as exc:
            print("FAIL", slug, exc)


main()
