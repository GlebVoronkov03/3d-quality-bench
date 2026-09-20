"""Скачивание 24 публичных эталонов с явной лицензией в DataBase/references/."""
from __future__ import annotations

import gzip
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "DataBase" / "references"
JACOBSON = "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data"

# slug -> (url, dest relative to references/, license note)
JOBS = [
    ("armadillo", f"{JACOBSON}/armadillo.obj", "academic_cg/armadillo.obj"),
    ("happy", f"{JACOBSON}/happy.obj", "academic_cg/happy.obj"),
    ("xyzrgb_dragon", f"{JACOBSON}/xyzrgb_dragon.obj", "academic_cg/xyzrgb_dragon.obj"),
    ("lucy", f"{JACOBSON}/lucy.obj", "academic_cg/lucy.obj"),
    ("max-planck", f"{JACOBSON}/max-planck.obj", "academic_cg/max-planck.obj"),
    ("bimba", f"{JACOBSON}/bimba.obj", "heritage_scan/bimba.obj"),
    ("igea", f"{JACOBSON}/igea.obj", "heritage_scan/igea.obj"),
    ("nefertiti", f"{JACOBSON}/nefertiti.obj", "heritage_scan/nefertiti.obj"),
    ("ogre", f"{JACOBSON}/ogre.obj", "human_character/ogre.obj"),
    ("homer", f"{JACOBSON}/homer.obj", "human_character/homer.obj"),
    ("cheburashka", f"{JACOBSON}/cheburashka.obj", "human_character/cheburashka.obj"),
    ("woody", f"{JACOBSON}/woody.obj", "human_character/woody.obj"),
    ("horse", f"{JACOBSON}/horse.obj", "organic/horse.obj"),
    ("spot", f"{JACOBSON}/spot.obj", "organic/spot.obj"),
    ("alligator", f"{JACOBSON}/alligator.obj", "organic/alligator.obj"),
    ("beast", f"{JACOBSON}/beast.obj", "organic/beast.obj"),
    ("fandisk", f"{JACOBSON}/fandisk.obj", "mechanical/fandisk.obj"),
    ("rocker-arm", f"{JACOBSON}/rocker-arm.obj", "mechanical/rocker-arm.obj"),
    ("beetle", f"{JACOBSON}/beetle.obj", "mechanical/beetle.obj"),
]

STANFORD_DRAGON_PLY = "http://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz"
FERTILITY_CANDIDATES = [
    "https://raw.githubusercontent.com/libigl/libigl-tutorial-data/master/fertility.off",
    "https://raw.githubusercontent.com/alecjacobson/geometry-processing-smoothing/master/data/fertility.off",
]
# Официальное зеркало GitHub Thingi10K → Hugging Face (не качаем полный tar.gz на 4–9 ГБ).
# Старые ID 11280/351334 в индексе Zhou & Jacobson отсутствуют.
THINGI_CANDIDATES = [
    ("thingi_a", "https://huggingface.co/datasets/Thingi10K/Thingi10K/resolve/main/raw_meshes/59228.stl", "mechanical/thingi_a.stl"),
    ("thingi_b", "https://huggingface.co/datasets/Thingi10K/Thingi10K/resolve/main/raw_meshes/90275.stl", "mechanical/thingi_b.stl"),
]
HUMAN_CANDIDATES = [
    # CC0-ish public domain body-ish meshes; first successful URL wins
    "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/woody.obj",  # fallback placeholder skipped if woody exists
]


def download(url: str, dest: Path, timeout: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"skip exists {dest}")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"GET {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "PLER-HQ-dataset-builder/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dest)
    print(f"  -> {dest} ({dest.stat().st_size} bytes)")


def convert_to_obj(src: Path, dest: Path) -> None:
    import trimesh

    dest.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load(src, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(g for g in mesh.geometry.values()))
    mesh.export(dest)
    print(f"converted {src.name} -> {dest}")


def download_stanford_dragon() -> None:
    dest_obj = REF / "academic_cg" / "stanford-dragon.obj"
    if dest_obj.exists() and dest_obj.stat().st_size > 1024:
        print("skip exists stanford-dragon")
        return
    tarball = REF / "academic_cg" / "_dragon_recon.tar.gz"
    try:
        download(STANFORD_DRAGON_PLY, tarball, timeout=300)
    except Exception as exc:
        print(f"Stanford dragon failed: {exc}")
        return
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        with tarfile.open(tarball, "r:gz") as tar:
            tar.extractall(td_path)
        ply_files = list(td_path.rglob("*.ply"))
        if not ply_files:
            print("no ply in dragon archive")
            return
        # prefer reconstructed mesh
        ply = sorted(ply_files, key=lambda p: p.stat().st_size, reverse=True)[0]
        convert_to_obj(ply, dest_obj)
    tarball.unlink(missing_ok=True)


def download_fertility() -> None:
    dest = REF / "heritage_scan" / "fertility.obj"
    if dest.exists() and dest.stat().st_size > 1024:
        print("skip exists fertility")
        return
    for url in FERTILITY_CANDIDATES:
        try:
            raw = REF / "heritage_scan" / "_fertility_src.off"
            download(url, raw)
            convert_to_obj(raw, dest)
            raw.unlink(missing_ok=True)
            return
        except Exception as exc:
            print(f"fertility candidate failed {url}: {exc}")
    print("fertility PENDING")


def download_thingi() -> None:
    for slug, url, rel in THINGI_CANDIDATES:
        stl = REF / rel
        obj = stl.with_suffix(".obj")
        try:
            download(url, stl, timeout=180)
            convert_to_obj(stl, obj)
        except Exception as exc:
            print(f"{slug} failed: {exc} (slot remains pending)")


def download_human_tpose() -> None:
    """Слот 49: Male anatomy figure (Sketchfab, CC BY). Файл кладётся вручную через официальный Download."""
    dest = REF / "human_character" / "male-anatomy.obj"
    if dest.exists() and dest.stat().st_size > 1024:
        print("skip exists male-anatomy")
        return
    print("male-anatomy: нужен официальный zip с Sketchfab (кнопка Download).")


def main() -> int:
    ok, fail = 0, 0
    for slug, url, rel in JOBS:
        dest = REF / rel
        try:
            download(url, dest)
            ok += 1
        except Exception as exc:
            print(f"FAIL {slug}: {exc}")
            fail += 1
    download_stanford_dragon()
    download_fertility()
    download_thingi()
    download_human_tpose()
    print(f"done jacobson ok={ok} fail={fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
