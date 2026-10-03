"""Landscapes: fuel, terrain and assets on a regular grid.

A landscape is three aligned rasters plus a cell size:

- ``fuel``       float in [0, 1]; 0 means non-burnable (water, rock).
- ``elevation``  metres, used for the slope effect on spread.
- ``assets``     bool; cells we want to protect (houses). Assets can burn but are never treated.

``synthetic_*`` landscapes are generated deterministically from a seed, so a fresh clone
reproduces exactly the same data. A real landscape (e.g. converted from LANDFIRE) can be
dropped in as ``data/landscape_<name>.npz`` with the same three arrays and ``cell_size_m``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.ndimage import gaussian_filter

from firelab import config

SYNTHETIC = {
    "synthetic_v1": {"size": 120, "seed": 7},     # 120 x 120 cells of 1 ha
    "synthetic_small": {"size": 60, "seed": 3},   # quick smoke tests
}


@dataclass(frozen=True, eq=False)
class Landscape:
    name: str
    fuel: np.ndarray
    elevation: np.ndarray
    assets: np.ndarray
    cell_size_m: float = 100.0

    @property
    def shape(self) -> tuple[int, int]:
        return self.fuel.shape

    @property
    def cell_ha(self) -> float:
        return self.cell_size_m ** 2 / 10_000

    @property
    def burnable(self) -> np.ndarray:
        return self.fuel > 0

    @property
    def treatable(self) -> np.ndarray:
        return self.burnable & ~self.assets

    @property
    def n_burnable(self) -> int:
        return int(self.burnable.sum())

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for a in (self.fuel, self.elevation, self.assets):
            h.update(np.ascontiguousarray(a).tobytes())
        h.update(str(self.cell_size_m).encode())
        return h.hexdigest()[:12]

    def summary(self) -> dict:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "cell_size_m": self.cell_size_m,
            "cell_ha": self.cell_ha,
            "burnable_cells": self.n_burnable,
            "asset_cells": int(self.assets.sum()),
            "total_area_ha": float(self.fuel.size * self.cell_ha),
            "fingerprint": self.fingerprint(),
        }


def _rescale(a: np.ndarray) -> np.ndarray:
    return (a - a.min()) / (a.max() - a.min() + 1e-12)


def _smooth_noise(rng: np.random.Generator, shape: tuple[int, int], sigma: float) -> np.ndarray:
    return gaussian_filter(rng.standard_normal(shape), sigma=sigma, mode="reflect")


def _disc(shape: tuple[int, int], center: tuple[float, float], radius: float) -> np.ndarray:
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius ** 2


def generate_synthetic(name: str = "synthetic_v1", size: int = 120, seed: int = 7,
                       cell_size_m: float = 100.0) -> Landscape:
    """Hilly, patchy landscape with a creek, rock on the ridge and two settlements.

    Row 0 is north and column 0 is west. The settlements sit in the east, downwind of the
    default south-westerly fire weather, so placement relative to wind matters.
    """
    rng = np.random.default_rng(seed)
    h = w = size
    yy, xx = np.mgrid[0:h, 0:w] / size

    hills = _rescale(_smooth_noise(rng, (h, w), sigma=size / 8))
    elev01 = _rescale(0.6 * hills + 0.4 * (xx + (1 - yy)) / 2)   # ridge rising to the north-east
    elevation = 200 + 900 * elev01

    coarse = _rescale(_smooth_noise(rng, (h, w), sigma=size / 15))
    fine = _rescale(_smooth_noise(rng, (h, w), sigma=1.5))
    fuel = np.clip(0.25 + 0.6 * coarse + 0.15 * fine, 0.05, 1.0)

    # A creek running south from the north edge (fire can go around its southern end)
    # and bare rock on the highest ground.
    creek_col = (0.30 + 0.06 * np.sin(2 * np.pi * 1.5 * yy[:, 0])) * w
    creek = (np.abs(np.arange(w)[None, :] - creek_col[:, None]) < 1.2) & (yy < 0.75)
    rock = elev01 > 0.93
    fuel[creek | rock] = 0.0

    assets = (_disc((h, w), (0.30 * h, 0.72 * w), 0.05 * size)
              | _disc((h, w), (0.62 * h, 0.80 * w), 0.035 * size))
    assets &= fuel > 0
    fuel[assets] = np.maximum(fuel[assets], 0.5)   # gardens and structures still carry fire

    return Landscape(name, fuel.astype(np.float32), elevation.astype(np.float32), assets, cell_size_m)


def _path(name: str):
    return config.DATA_DIR / f"landscape_{name}.npz"


def save_landscape(ls: Landscape) -> None:
    config.ensure_dirs()
    np.savez_compressed(_path(ls.name), fuel=ls.fuel, elevation=ls.elevation,
                        assets=ls.assets, cell_size_m=ls.cell_size_m)


@lru_cache(maxsize=4)
def load_landscape(name: str | None = None) -> Landscape:
    """Load ``data/landscape_<name>.npz``; synthetic landscapes are generated on first use."""
    name = name or config.LANDSCAPE
    path = _path(name)
    if path.exists():
        with np.load(path) as z:
            ls = Landscape(name, z["fuel"].astype(np.float32), z["elevation"].astype(np.float32),
                           z["assets"].astype(bool), float(z["cell_size_m"]))
    elif name in SYNTHETIC:
        ls = generate_synthetic(name, **SYNTHETIC[name])
        save_landscape(ls)
    else:
        raise FileNotFoundError(f"No landscape file {path} and '{name}' is not a known synthetic landscape.")
    for a in (ls.fuel, ls.elevation, ls.assets):
        a.flags.writeable = False   # shared via the cache; never mutate
    return ls
