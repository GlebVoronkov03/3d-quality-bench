"""Очистка __pycache__ и перезагрузка локальных модулей."""

from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path
from typing import Iterable


def clear_pycache(root: Path | str | None = None) -> int:
    """Удаляет все каталоги __pycache__ под root. Возвращает число удалённых."""
    base = Path(root) if root else Path(__file__).resolve().parent
    removed = 0
    for cache_dir in base.rglob("__pycache__"):
        try:
            shutil.rmtree(cache_dir)
            removed += 1
        except OSError:
            pass
    return removed


def reload_local_modules(module_names: Iterable[str]) -> None:
    """Перезагружает локальные модули из sys.modules (после очистки кэша)."""
    for name in module_names:
        if name in sys.modules:
            importlib.reload(sys.modules[name])
