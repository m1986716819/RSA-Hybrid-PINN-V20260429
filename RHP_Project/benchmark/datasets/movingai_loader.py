from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .scene import Scene

_FREE_CHARS = frozenset({".", "G"})
_OBSTACLE_CHARS = frozenset({"@", "O", "T", "S", "W"})
_ALLOWED = _FREE_CHARS | _OBSTACLE_CHARS


def parse_map_file(map_path: str) -> Tuple[np.ndarray, Dict[str, int]]:
    """Parse a MovingAI .map file into a binary occupancy grid.

    Returns:
        grid: (H, W) uint8 array, 0=free, 1=obstacle
        meta: dict with height, width
    """
    with open(map_path, "r") as f:
        lines = [line.rstrip("\n\r") for line in f]

    header_idx = 0
    meta: Dict[str, int] = {}
    while header_idx < len(lines):
        line = lines[header_idx].strip()
        if not line:
            header_idx += 1
            continue
        if line.lower().startswith("type"):
            header_idx += 1
            continue
        if line.lower().startswith("height"):
            meta["height"] = int(line.split()[1])
            header_idx += 1
            continue
        if line.lower().startswith("width"):
            meta["width"] = int(line.split()[1])
            header_idx += 1
            continue
        if line.lower() == "map":
            header_idx += 1
            break
        header_idx += 1

    h = int(meta.get("height", 0))
    w = int(meta.get("width", 0))
    if h <= 0 or w <= 0:
        raise ValueError(f"Could not parse .map header from {map_path}")

    grid = np.zeros((h, w), dtype=np.uint8)
    row = 0
    while header_idx < len(lines) and row < h:
        line = lines[header_idx].rstrip()
        header_idx += 1
        if len(line) < w:
            line = line + " " * (w - len(line))
        for col, ch in enumerate(line[:w]):
            if ch in _OBSTACLE_CHARS:
                grid[row, col] = 1
            elif ch not in _FREE_CHARS:
                grid[row, col] = 1
        row += 1

    return grid, meta


def parse_scen_file(scen_path: str) -> List[Dict]:
    """Parse a MovingAI .scen file.

    Returns:
        List of dicts with keys:
            bucket, map_name, map_w, map_h,
            start_x, start_y, goal_x, goal_y, optimal_length
        Coordinates are in (col, row) = (x, y) per MovingAI convention.
    """
    scenes: List[Dict] = []
    with open(scen_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("version"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            record = {
                "bucket": int(parts[0]),
                "map_name": parts[1],
                "map_w": int(parts[2]),
                "map_h": int(parts[3]),
                "start_x": int(parts[4]),
                "start_y": int(parts[5]),
                "goal_x": int(parts[6]),
                "goal_y": int(parts[7]),
                "optimal_length": float(parts[8]) if len(parts) > 8 else None,
            }
            scenes.append(record)
    return scenes


class MovingAILoader:
    """Load MovingAI benchmarks and yield Scene objects.

    Usage:
        loader = MovingAILoader("/path/to/movingai_data")
        scenes = loader.load("maze512", max_scenes=50)
        # or
        scene = loader.load_single("maze512", scen_idx=0)
    """

    FREE = 0
    OBSTACLE = 1

    def __init__(self, data_root: str):
        self.data_root = Path(data_root).resolve()
        if not self.data_root.is_dir():
            raise NotADirectoryError(f"MovingAI data root not found: {data_root}")

    def _resolve_map(self, map_name: str) -> str:
        candidates = [
            self.data_root / f"{map_name}.map",
            self.data_root / map_name / f"{map_name}.map",
            self.data_root / f"{map_name}.map" / f"{map_name}.map",
        ]
        for c in candidates:
            if c.is_file():
                return str(c)
        name_stem = Path(map_name).stem
        for p in self.data_root.rglob(f"{name_stem}.map"):
            return str(p)
        raise FileNotFoundError(f"Cannot find .map file for: {map_name} in {self.data_root}")

    def _resolve_scen(self, map_name: str) -> Optional[str]:
        candidates = [
            self.data_root / f"{map_name}.scen",
            self.data_root / map_name / f"{map_name}.scen",
        ]
        for c in candidates:
            if c.is_file():
                return str(c)
        name_stem = Path(map_name).stem
        for p in self.data_root.rglob(f"{name_stem}.scen"):
            return str(p)
        return None

    def load_map_only(self, map_name: str) -> Scene:
        """Load only the map with no start/goal (for procedural placement)."""
        map_path = self._resolve_map(map_name)
        grid, meta = parse_map_file(map_path)
        return Scene(
            grid=grid,
            start_px=(0, 0),
            goal_px=(0, 0),
            name=map_name,
            map_name=map_name,
            metadata={"map_path": map_path, **meta},
        )

    def load(self, map_name: str, max_scenes: Optional[int] = None) -> List[Scene]:
        """Load all scenes for a given map from .map + .scen files."""
        map_path = self._resolve_map(map_name)
        grid, meta = parse_map_file(map_path)
        h, w = meta["height"], meta["width"]

        scen_path = self._resolve_scen(map_name)
        if scen_path is None:
            return [Scene(
                grid=grid, start_px=(0, 0), goal_px=(0, 0),
                name=map_name, map_name=map_name,
                metadata={"map_path": map_path, **meta},
            )]

        records = parse_scen_file(scen_path)
        if max_scenes is not None and max_scenes > 0:
            records = records[:max_scenes]

        scenes: List[Scene] = []
        for i, rec in enumerate(records):
            if rec["map_w"] != w or rec["map_h"] != h:
                continue
            start_px = (int(rec["start_y"]), int(rec["start_x"]))
            goal_px = (int(rec["goal_y"]), int(rec["goal_x"]))
            if not (0 <= start_px[0] < h and 0 <= start_px[1] < w):
                start_px = (max(0, min(h - 1, start_px[0])),
                            max(0, min(w - 1, start_px[1])))
            if not (0 <= goal_px[0] < h and 0 <= goal_px[1] < w):
                goal_px = (max(0, min(h - 1, goal_px[0])),
                           max(0, min(w - 1, goal_px[1])))
            scene = Scene(
                grid=grid.copy(),
                start_px=start_px,
                goal_px=goal_px,
                optimal_length=rec.get("optimal_length"),
                name=f"{map_name}_{i:04d}",
                map_name=map_name,
                metadata={"bucket": rec["bucket"], "scen_index": i,
                          "map_path": map_path, "scen_path": scen_path, **meta},
            )
            scenes.append(scene)
        return scenes

    @staticmethod
    def list_available(data_root: str) -> List[str]:
        root = Path(data_root)
        names: List[str] = []
        for f in sorted(root.rglob("*.map")):
            rel = f.relative_to(root)
            names.append(str(rel.with_suffix("")))
        return names

    @staticmethod
    def download_map(map_name: str, save_dir: str) -> str:
        """Download a single MovingAI map by name from the official repository.

        Uses the MovingAI benchmark file server.

        Args:
            map_name: e.g. 'maze512', 'arena', 'room32'
            save_dir: directory to save the downloaded files

        Returns:
            Path to the downloaded .map file.
        """
        import urllib.request
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        base_url = f"https://movingai.com/benchmarks"
        map_url = f"{base_url}/{map_name}/{map_name}.map"
        map_dst = save_path / f"{map_name}.map"

        if not map_dst.exists():
            print(f"  [download] {map_url} -> {map_dst}")
            urllib.request.urlretrieve(map_url, str(map_dst))

        scen_url = f"{base_url}/{map_name}/{map_name}.scen"
        scen_dst = save_path / f"{map_name}.scen"
        if not scen_dst.exists():
            try:
                print(f"  [download] {scen_url} -> {scen_dst}")
                urllib.request.urlretrieve(scen_url, str(scen_dst))
            except Exception:
                print(f"  [download] .scen not found, skipping: {scen_url}")

        return str(map_dst)
