from typing import Optional, Union

import numpy as np

from arraycontext import ArrayContext
from pytential.source import PointPotentialSource
from pytential.target import PointsTarget


def as_source(actx: ArrayContext, points: np.ndarray) -> PointPotentialSource:
    return PointPotentialSource(actx.from_numpy(points))


def as_target(actx: ArrayContext, points: np.ndarray) -> PointsTarget:
    return PointsTarget(actx.from_numpy(points))


# {{{ point clouds

def make_grid_points(
        nx: int, ny: int, nz: Optional[int] = None,
        ) -> np.ndarray:
    """Construct points on an equidistant grid :math:`[-0.5, 0.5]^d`."""
    ambient_dim = 2 if nz is None else 3

    if ambient_dim == 2:
        points = np.mgrid[-0.5:0.5:1j*nx, -0.5:0.5:1j*ny]
    elif ambient_dim == 3:
        points = np.mgrid[-0.5:0.5:1j*nx, -0.5:0.5:1j*ny, -0.5:0.5:1j*nz]
    else:
        raise ValueError(f"unsupported dimension: {ambient_dim}")

    return points.reshape(points.shape[0], -1)


def make_random_points(
        ambient_dim: int, npoints: int, *, random: str = "uniform",
        ) -> np.ndarray:
    rng = np.random.default_rng()
    if random == "uniform":
        return -0.5 + rng.random(size=(ambient_dim, npoints))
    elif random == "normal":
        # NOTE: multiplying by 0.2 seems to make it sufficiently likely that
        # all the values are in [-0.5, 0.5] for practical purposes
        return 0.2 * rng.standard_normal(size=(ambient_dim, npoints))
    else:
        raise ValueError(f"unknown random distribution: '{random}'")


def make_axis_points(
        axis: np.ndarray, npoints: int,
        ) -> np.ndarray:
    ambient_dim, = axis.shape

    t = np.linspace(-0.5, 0.5, npoints)
    return t * axis.reshape(-1, 1)


def make_circle(npoints: int) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, npoints)
    return np.stack([np.cos(theta), np.sin(theta)])

# }}}


# {{{ maps

def affine_map(x: np.ndarray, *,
        mat: Optional[Union[np.ndarray, float]] = None,
        b: Optional[Union[np.ndarray, float]] = None) -> np.ndarray:
    from numbers import Number
    y = x.copy()
    if mat is not None:
        if isinstance(mat, Number):
            y = mat * y
        else:
            y = mat @ y

    if b is not None:
        if isinstance(b, Number):
            y = y + b
        else:
            y = y + b.reshape(-1, 1)

    return y

# }}}
