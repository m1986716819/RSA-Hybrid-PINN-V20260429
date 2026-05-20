from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class RSAResult:
    t: np.ndarray
    prev_i: np.ndarray
    prev_j: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    start_ij: Tuple[int, int]
    goal_ij: Tuple[int, int]

    def interpolate_t(self, xy: np.ndarray) -> np.ndarray:
        x = xy[:, 0]
        y = xy[:, 1]
        w = self.xs.shape[0]
        h = self.ys.shape[0]

        x = np.clip(x, self.xs[0], self.xs[-1])
        y = np.clip(y, self.ys[0], self.ys[-1])

        j = np.searchsorted(self.xs, x) - 1
        i = np.searchsorted(self.ys, y) - 1
        j = np.clip(j, 0, w - 2)
        i = np.clip(i, 0, h - 2)

        x0 = self.xs[j]
        x1 = self.xs[j + 1]
        y0 = self.ys[i]
        y1 = self.ys[i + 1]
        tx = (x - x0) / (x1 - x0 + 1e-12)
        ty = (y - y0) / (y1 - y0 + 1e-12)

        t00 = self.t[i, j]
        t10 = self.t[i, j + 1]
        t01 = self.t[i + 1, j]
        t11 = self.t[i + 1, j + 1]
        finite = np.isfinite(self.t)
        if np.any(finite):
            fill = float(np.max(self.t[finite]))
        else:
            fill = 1.0
        t00 = np.where(np.isfinite(t00), t00, fill)
        t10 = np.where(np.isfinite(t10), t10, fill)
        t01 = np.where(np.isfinite(t01), t01, fill)
        t11 = np.where(np.isfinite(t11), t11, fill)
        t0 = t00 * (1 - tx) + t10 * tx
        t1 = t01 * (1 - tx) + t11 * tx
        return t0 * (1 - ty) + t1 * ty

    def backtrack_path_xy(self, max_len: int = 10000) -> np.ndarray:
        i, j = self.goal_ij
        si, sj = self.start_ij
        pts: List[Tuple[float, float]] = []
        h, w = self.t.shape

        for _ in range(max_len):
            x = float(self.xs[j])
            y = float(self.ys[i])
            pts.append((x, y))
            if (i, j) == (si, sj):
                break
            pi = int(self.prev_i[i, j])
            pj = int(self.prev_j[i, j])
            if pi < 0 or pj < 0 or pi >= h or pj >= w or (pi == i and pj == j):
                break
            i, j = pi, pj
        pts.reverse()
        return np.asarray(pts, dtype=np.float32)


class RSAEngine:
    def __init__(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        connectivity: Literal[4, 8] = 8,
        speed_eps: float = 1e-6,
    ) -> None:
        self.xs = xs.astype(np.float32)
        self.ys = ys.astype(np.float32)
        self.connectivity: Literal[4, 8] = connectivity
        self.speed_eps = float(speed_eps)

        if self.connectivity == 4:
            self._nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        elif self.connectivity == 8:
            self._nbrs = [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]
        else:
            raise ValueError("connectivity must be 4 or 8")

    @staticmethod
    def _nearest_ij(xs: np.ndarray, ys: np.ndarray, xy: Tuple[float, float]) -> Tuple[int, int]:
        x, y = xy
        j = int(np.argmin(np.abs(xs - x)))
        i = int(np.argmin(np.abs(ys - y)))
        return i, j

    def solve(
        self,
        speed: np.ndarray,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
    ) -> RSAResult:
        speed = speed.astype(np.float32)
        h, w = speed.shape
        if h != self.ys.shape[0] or w != self.xs.shape[0]:
            raise ValueError("speed shape must match (len(ys), len(xs))")

        start_ij = self._nearest_ij(self.xs, self.ys, start_xy)
        goal_ij = self._nearest_ij(self.xs, self.ys, goal_xy)

        t = np.full((h, w), np.inf, dtype=np.float32)
        prev_i = np.full((h, w), -1, dtype=np.int32)
        prev_j = np.full((h, w), -1, dtype=np.int32)
        visited = np.zeros((h, w), dtype=np.bool_)

        si, sj = start_ij
        if speed[si, sj] <= 0.0:
            raise ValueError("Start is inside obstacle or zero-speed region.")

        t[si, sj] = 0.0
        prev_i[si, sj] = si
        prev_j[si, sj] = sj

        pq: List[Tuple[float, int, int]] = [(0.0, si, sj)]
        while pq:
            cur_t, i, j = heapq.heappop(pq)
            if visited[i, j]:
                continue
            visited[i, j] = True
            if (i, j) == goal_ij:
                break

            for di, dj in self._nbrs:
                ni = i + di
                nj = j + dj
                if ni < 0 or nj < 0 or ni >= h or nj >= w:
                    continue
                if visited[ni, nj]:
                    continue
                if speed[ni, nj] <= 0.0:
                    continue

                dx = float(self.xs[nj] - self.xs[j])
                dy = float(self.ys[ni] - self.ys[i])
                ds = float(np.hypot(dx, dy))
                f_ij = 0.5 * (float(speed[i, j]) + float(speed[ni, nj]))
                f_ij = max(self.speed_eps, f_ij)
                cand = cur_t + ds / f_ij
                if cand < t[ni, nj]:
                    t[ni, nj] = cand
                    prev_i[ni, nj] = i
                    prev_j[ni, nj] = j
                    heapq.heappush(pq, (float(cand), ni, nj))

        return RSAResult(
            t=t,
            prev_i=prev_i,
            prev_j=prev_j,
            xs=self.xs,
            ys=self.ys,
            start_ij=start_ij,
            goal_ij=goal_ij,
        )

    @staticmethod
    def wavefront_weight(
        t: np.ndarray,
        free_mask: np.ndarray,
        base_weight: float = 0.2,
        wavefront_power: float = 1.0,
        obstacle_band: float = 0.0,
        xs: Optional[np.ndarray] = None,
        ys: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        t = t.astype(np.float32)
        mask = free_mask.astype(np.float32)
        finite = np.isfinite(t) & (free_mask.astype(bool))
        if not np.any(finite):
            return np.full_like(t, base_weight, dtype=np.float32)

        tt = np.where(finite, t, np.nan)
        gx = np.nan_to_num(np.gradient(tt, axis=1), nan=0.0)
        gy = np.nan_to_num(np.gradient(tt, axis=0), nan=0.0)
        lap = np.nan_to_num(np.gradient(gx, axis=1) + np.gradient(gy, axis=0), nan=0.0)
        w = np.abs(lap) ** float(wavefront_power)

        w = w * mask
        w = w / (np.percentile(w[finite], 95) + 1e-12)
        w = np.clip(w, 0.0, 1.0)
        w = base_weight + (1.0 - base_weight) * w

        if obstacle_band > 0.0 and xs is not None and ys is not None:
            dx = float(xs[1] - xs[0]) if xs.shape[0] > 1 else 1.0
            dy = float(ys[1] - ys[0]) if ys.shape[0] > 1 else 1.0
            band_cells = int(np.ceil(obstacle_band / max(1e-12, min(dx, dy))))
            occ = (~free_mask).astype(np.uint8)
            band = occ.copy()
            for _ in range(max(0, band_cells)):
                band = np.maximum.reduce(
                    [
                        np.pad(band[1:, :], ((0, 1), (0, 0))),
                        np.pad(band[:-1, :], ((1, 0), (0, 0))),
                        np.pad(band[:, 1:], ((0, 0), (0, 1))),
                        np.pad(band[:, :-1], ((0, 0), (1, 0))),
                    ]
                )
            band = band.astype(bool) & free_mask.astype(bool)
            w = np.where(band, np.maximum(w, 0.8), w)

        return w.astype(np.float32)


@dataclass(frozen=True)
class GatewayInfo:
    gate_xy: np.ndarray
    d_target: np.ndarray
    d_targets: np.ndarray
    indices: np.ndarray


def extract_gateway_segment(
    path_xy: np.ndarray,
    sdf_values: np.ndarray,
    segment_len: int = 16,
    sdf_weight: float = 1.0,
    diff_weight: float = 1.0,
    min_free_sdf: float = 0.0,
) -> GatewayInfo:
    path_xy = np.asarray(path_xy, dtype=np.float32)
    sdf_values = np.asarray(sdf_values, dtype=np.float32).reshape(-1)
    n = path_xy.shape[0]
    if n < 3:
        raise ValueError("path_xy must have at least 3 points.")
    if sdf_values.shape[0] != n:
        raise ValueError("sdf_values must have same length as path_xy.")

    sdf = np.maximum(sdf_values, float(min_free_sdf))
    ds = np.linalg.norm(path_xy[1:] - path_xy[:-1], axis=1) + 1e-12
    dsdf = np.abs(np.diff(sdf_values)) / ds
    dsdf = np.concatenate([dsdf[:1], dsdf], axis=0)

    score = float(sdf_weight) * (1.0 / (sdf + 1e-6)) + float(diff_weight) * dsdf
    i_max = int(np.argmax(score))

    L = int(max(4, segment_len))
    i1 = max(0, i_max - L // 2)
    i2 = min(n, i1 + L)
    i1 = max(0, i2 - L)
    idx = np.arange(i1, i2, dtype=np.int32)
    gate_xy = path_xy[idx]

    lookahead = max(3, min(40, n - 1))
    d_list = []
    for i in idx:
        j = min(n - 1, int(i + lookahead))
        d = path_xy[j] - path_xy[i]
        dn = float(np.linalg.norm(d) + 1e-12)
        d_list.append(d / dn)
    d_targets = np.stack(d_list, axis=0).astype(np.float32)
    d_target = np.mean(np.stack(d_list, axis=0), axis=0)
    d_target = d_target / (float(np.linalg.norm(d_target) + 1e-12))

    return GatewayInfo(
        gate_xy=gate_xy,
        d_target=d_target.astype(np.float32),
        d_targets=d_targets,
        indices=idx,
    )


def sample_gate_band(
    gate_xy: np.ndarray,
    d_targets: np.ndarray,
    radius: float = 0.15,
    num_samples: int = 256,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    gate_xy = np.asarray(gate_xy, dtype=np.float32)
    d_targets = np.asarray(d_targets, dtype=np.float32)
    if gate_xy.ndim != 2 or gate_xy.shape[1] != 2:
        raise ValueError("gate_xy must have shape [M,2].")
    if d_targets.ndim != 2 or d_targets.shape != gate_xy.shape:
        raise ValueError("d_targets must have shape [M,2] matching gate_xy.")
    if gate_xy.shape[0] < 1:
        raise ValueError("gate_xy must contain at least 1 point.")
    if int(num_samples) < 1:
        raise ValueError("num_samples must be >= 1.")

    if rng is None:
        rng = np.random.default_rng(0)

    m = gate_xy.shape[0]
    idx = rng.integers(0, m, size=(int(num_samples),), endpoint=False)
    base = gate_xy[idx]
    base_d = d_targets[idx]
    dn = np.linalg.norm(base_d, axis=1, keepdims=True) + 1e-12
    d_out = (base_d / dn).astype(np.float32)

    # Build a narrow capsule aligned with the RSA crossing direction so the
    # gate constraint covers a short tunnel instead of an isotropic disk.
    tangent = d_out
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1).astype(np.float32)
    axial_half = float(radius) * 1.15
    lateral_half = float(radius) * 0.60
    axial = rng.uniform(-axial_half, axial_half, size=(int(num_samples), 1)).astype(np.float32)
    lateral = rng.uniform(-lateral_half, lateral_half, size=(int(num_samples), 1)).astype(np.float32)
    jitter = rng.normal(loc=0.0, scale=float(radius) * 0.04, size=(int(num_samples), 2)).astype(np.float32)
    pts = (base + axial * tangent + lateral * normal + jitter).astype(np.float32)
    return pts, d_out
