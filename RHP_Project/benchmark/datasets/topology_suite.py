from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from RHP_Project.envs.maze_2d import Bounds2D, Maze2DEnv

from .scene import Scene


def _env_to_scene(
    env: Maze2DEnv,
    start_world: Tuple[float, float],
    goal_world: Tuple[float, float],
    grid_size: Tuple[int, int],
    name: str = "",
) -> Scene:
    """Convert a Maze2DEnv to a Scene via grid discretisation."""
    h, w = int(grid_size[0]), int(grid_size[1])
    xs = np.linspace(env.bounds.x_min, env.bounds.x_max, w, dtype=np.float32)
    ys = np.linspace(env.bounds.y_min, env.bounds.y_max, h, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    xy = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)
    with torch.no_grad():
        sdf_vals = env.sdf(torch.from_numpy(xy)).numpy().reshape(h, w)
    grid = (sdf_vals <= 0.0).astype(np.uint8)

    def _world_to_pixel(wx: float, wy: float) -> Tuple[int, int]:
        col = int(round((wx - env.bounds.x_min) / (env.bounds.x_max - env.bounds.x_min) * float(w - 1)))
        row = int(round((wy - env.bounds.y_min) / (env.bounds.y_max - env.bounds.y_min) * float(h - 1)))
        return (max(0, min(h - 1, row)), max(0, min(w - 1, col)))

    start_px = _world_to_pixel(start_world[0], start_world[1])
    goal_px = _world_to_pixel(goal_world[0], goal_world[1])

    return Scene(grid=grid, start_px=start_px, goal_px=goal_px, name=name)


class TopologySuite:
    """Generate procedural benchmark scenes with known topological challenges.

    Each method returns a list of Scene objects. All scenes are in [0,1]^2
    world coordinates with configurable grid resolution.
    """

    DEFAULT_GRID: Tuple[int, int] = (100, 100)

    def __init__(self, grid_size: Optional[Tuple[int, int]] = None):
        self.grid_size = grid_size or self.DEFAULT_GRID
        self.bounds = Bounds2D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0)

    def u_shape_trap(self,
                     wall_thickness: float = 0.03,
                     left_x: float = 0.25,
                     u_depth: float = 0.35,
                     u_height: float = 0.55,
                     start: Tuple[float, float] = (0.08, 0.08),
                     goal: Tuple[float, float] = (0.9, 0.9),
                     num_variants: int = 1) -> List[Scene]:
        scenes = []
        for i in range(num_variants):
            offset = float(i) * 0.02
            env = Maze2DEnv.make_trap_u_shape(
                bounds=self.bounds,
                wall_thickness=wall_thickness,
                left_x=left_x + offset * 0.5,
                u_depth=u_depth,
                u_height=u_height + offset * 0.3,
                center_y=0.5,
                device="cpu",
            )
            scenes.append(_env_to_scene(env, start, goal, self.grid_size,
                                        name=f"u_shape_trap_v{i}"))
        return scenes

    def narrow_passage(self,
                       passage_width: float = 0.12,
                       block_width: float = 0.9,
                       block_height: float = 0.35,
                       start: Tuple[float, float] = (0.08, 0.08),
                       goal: Tuple[float, float] = (0.9, 0.9),
                       num_variants: int = 1) -> List[Scene]:
        scenes = []
        for i in range(num_variants):
            pw = passage_width * (1.0 - float(i) * 0.12)
            pw = max(pw, 0.04)
            env = Maze2DEnv.make_narrow_passage(
                bounds=self.bounds,
                passage_width=pw,
                block_width=block_width,
                block_height=block_height,
                device="cpu",
            )
            scenes.append(_env_to_scene(env, start, goal, self.grid_size,
                                        name=f"narrow_passage_v{i}_w{pw:.2f}"))
        return scenes

    def bug_trap(self,
                 start: Tuple[float, float] = (0.08, 0.08),
                 goal: Tuple[float, float] = (0.9, 0.9),
                 num_variants: int = 1) -> List[Scene]:
        """Bug trap: an enclosure with a small exit opening."""
        scenes = []
        for i in range(num_variants):
            gap = 0.10 + float(i) * 0.02
            trap_left = 0.65
            trap_bottom = 0.15
            trap_top = 0.85
            trap_right = 0.95

            obstacles = [
                dict(kind="box", center=((trap_left + trap_right) * 0.5, trap_bottom),
                     half_size=((trap_right - trap_left) * 0.5, 0.02)),
                dict(kind="box", center=((trap_left + trap_right) * 0.5, trap_top),
                     half_size=((trap_right - trap_left) * 0.5, 0.02)),
                dict(kind="box", center=(trap_left, (trap_bottom + trap_top) * 0.5),
                     half_size=(0.02, (trap_top - trap_bottom) * 0.5)),
                dict(kind="box", center=(trap_right, (trap_bottom + trap_top) * 0.5),
                     half_size=(0.02, (trap_top - trap_bottom - gap) * 0.5)),
            ]
            env = Maze2DEnv(bounds=self.bounds, obstacles=obstacles, device="cpu")
            scenes.append(_env_to_scene(env, start, goal, self.grid_size,
                                        name=f"bug_trap_v{i}"))
        return scenes

    def maze_corridor(self,
                      num_turns: int = 3,
                      corridor_width: float = 0.10,
                      start: Tuple[float, float] = (0.05, 0.05),
                      goal: Tuple[float, float] = (0.9, 0.9),
                      num_variants: int = 1) -> List[Scene]:
        """Multi-turn corridor maze."""
        scenes = []
        for i in range(num_variants):
            obstacles = []
            hw = corridor_width * 0.5
            n = int(num_turns)
            seg_len = 0.9 / float(max(n + 1, 1))
            x, y = 0.05, 0.05
            turn_horizontal = True

            for t in range(n):
                if turn_horizontal:
                    x_next = x + seg_len
                    cx = (x + x_next) * 0.5
                    cy = y - hw
                    obstacles.append(dict(kind="box", center=(cx, cy),
                                          half_size=((x_next - x) * 0.5, hw)))
                    x = x_next
                else:
                    y_next = y + seg_len
                    cx = x + hw
                    cy = (y + y_next) * 0.5
                    obstacles.append(dict(kind="box", center=(cx, cy),
                                          half_size=(hw, (y_next - y) * 0.5)))
                    y = y_next
                turn_horizontal = not turn_horizontal

            env = Maze2DEnv(bounds=self.bounds, obstacles=obstacles, device="cpu")
            scenes.append(_env_to_scene(env, start, goal, self.grid_size,
                                        name=f"maze_corridor_v{i}_turns{num_turns}"))
        return scenes

    def non_convex_obstacle(self,
                            start: Tuple[float, float] = (0.08, 0.08),
                            goal: Tuple[float, float] = (0.9, 0.9),
                            num_variants: int = 1) -> List[Scene]:
        """L-shaped non-convex obstacle that creates local minima for gradient methods."""
        scenes = []
        for i in range(num_variants):
            lx = 0.35 + float(i) * 0.03
            ly = 0.35 + float(i) * 0.03
            obstacles = [
                dict(kind="box", center=(lx, 0.5), half_size=(0.03, 0.2)),
                dict(kind="box", center=(0.5, ly), half_size=(0.2, 0.03)),
            ]
            env = Maze2DEnv(bounds=self.bounds, obstacles=obstacles, device="cpu")
            scenes.append(_env_to_scene(env, start, goal, self.grid_size,
                                        name=f"non_convex_v{i}"))
        return scenes

    def open_space(self,
                   start: Tuple[float, float] = (0.08, 0.08),
                   goal: Tuple[float, float] = (0.9, 0.9)) -> List[Scene]:
        env = Maze2DEnv.make_open_space(bounds=self.bounds, device="cpu")
        return [_env_to_scene(env, start, goal, self.grid_size, name="open_space")]

    def generate_all(self, variants: int = 1) -> Dict[str, List[Scene]]:
        return {
            "u_shape_trap": self.u_shape_trap(num_variants=variants),
            "narrow_passage": self.narrow_passage(num_variants=variants),
            "bug_trap": self.bug_trap(num_variants=variants),
            "maze_corridor": self.maze_corridor(num_variants=variants),
            "non_convex_obstacle": self.non_convex_obstacle(num_variants=variants),
            "open_space": self.open_space(),
        }
