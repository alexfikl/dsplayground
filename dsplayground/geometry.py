from typing import Optional, Union

import numpy as np
import numpy.linalg as la


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
        points = np.mgrid[-0.5:0.5:1j*nx, -0.5:0.5:1j*ny]        # type: ignore[misc]
    elif ambient_dim == 3:
        assert nz is not None
        points = np.mgrid[
                -0.5:0.5:1j*nx, -0.5:0.5:1j*ny, -0.5:0.5:1j*nz]  # type: ignore[misc]
    else:
        raise ValueError(f"unsupported dimension: {ambient_dim}")

    return points.reshape(points.shape[0], -1)


def make_random_points_in_box(ambient_dim: int, npoints: int) -> np.ndarray:
    rng = np.random.default_rng()
    return -0.5 + rng.random(size=(ambient_dim, npoints))


def make_random_points_in_sphere(
        ambient_dim: int, npoints: int, *,
        rmin: float = 0.0, rmax: float = 1.0,
        tmin: float = 0.0, tmax: float = 2.0 * np.pi,
        pmin: float = 0.0, pmax: float = np.pi,
        ) -> np.ndarray:
    assert rmin >= 0.0 and rmax > rmin
    assert tmin >= 0.0 and tmax <= 2.0 * np.pi and tmin < tmax

    rng = np.random.default_rng()
    theta = tmin + (tmax - tmin) * rng.random(size=npoints)
    r = rmin + (rmax - rmin) * rng.random(size=npoints)

    if ambient_dim == 2:
        return np.stack([
            r * np.cos(theta),
            r * np.sin(theta)
            ])
    elif ambient_dim == 3:
        assert pmin >= 0.0 and pmax <= np.pi and pmin < pmax

        phi = pmin + (pmax - pmin) * rng.random(size=npoints)
        return np.stack([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi)
            ])
    else:
        raise ValueError(f"unsupported dimension: {ambient_dim}")


def make_axis_points(
        axis: np.ndarray, npoints: int,
        ) -> np.ndarray:
    t = np.linspace(-0.5, 0.5, npoints)
    return t * axis.reshape(-1, 1)


def make_circle(npoints: int, *, endpoint: bool = False) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, npoints, endpoint=endpoint)
    return np.stack([np.cos(theta), np.sin(theta)])


def make_sphere(npoints: int, *, method: str = "equidistant") -> np.ndarray:
    if method == "fibonacci":
        from pytools import sphere_sample_fibonacci
        return sphere_sample_fibonacci(npoints, r=1.0, optimize=None)
    elif method == "equidistant":
        from pytools import sphere_sample_equidistant
        return sphere_sample_equidistant(npoints, r=1.0)
    else:
        raise ValueError(f"unknown sampling method: '{method}'")
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
            assert isinstance(mat, np.ndarray)
            y = mat @ y

    if b is not None:
        if isinstance(b, Number):
            y = y + b
        else:
            assert isinstance(b, np.ndarray)
            y = y + b.reshape(-1, 1)

    return y

# }}}


# {{{

def get_point_radius_and_center(points: np.ndarray) -> Union[float, np.ndarray]:
    center = np.mean(points, axis=1)
    radius = np.max(la.norm(points - center.reshape(-1, 1), ord=2, axis=0))

    return radius, center

# }}}
