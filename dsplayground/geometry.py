from typing import Any, Optional, Tuple, Union

import numpy as np
import numpy.linalg as la

from arraycontext import ArrayContext
from meshmode.mesh import Mesh
from meshmode.discretization import Discretization
from pytools import memoize_on_first_arg

from pytential.source import PointPotentialSource
from pytential.target import PointsTarget
from pytential.linalg import IndexList


# pylint: disable-next=abstract-method
class ExpansionPointPotentialSource(PointPotentialSource):
    def __init__(self, nodes: Any, qbx_order: int) -> None:
        super().__init__(nodes)
        self.qbx_order = qbx_order

    def get_expansion_for_qbx_direct_eval(self, base_kernel, target_kernels):
        from sumpy.expansion.local import LineTaylorLocalExpansion
        from sumpy.kernel import TargetDerivativeRemover

        txr = TargetDerivativeRemover()
        if any(knl != txr(knl) for knl in target_kernels):
            raise ValueError

        return LineTaylorLocalExpansion(base_kernel, self.qbx_order)


def as_source(
        actx: ArrayContext, points: np.ndarray, *,
        qbx_order: Optional[int] = None) -> PointPotentialSource:
    points = actx.from_numpy(points)
    if qbx_order is None:
        return PointPotentialSource(points)
    else:
        return ExpansionPointPotentialSource(points, qbx_order)


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
    assert 0.0 <= rmin < rmax
    assert 0.0 <= tmin < tmax <= 2.0 * np.pi

    rng = np.random.default_rng()
    theta = tmin + (tmax - tmin) * rng.random(size=npoints)
    r = rmin + (rmax - rmin) * rng.random(size=npoints)

    if ambient_dim == 2:
        return np.stack([
            r * np.cos(theta),
            r * np.sin(theta)
            ])
    elif ambient_dim == 3:
        assert 0.0 <= pmin < pmax <= np.pi

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
        return sphere_sample_fibonacci(npoints, r=1.0, optimize="average")
    elif method == "equidistant":
        from pytools import sphere_sample_equidistant
        return sphere_sample_equidistant(npoints, r=1.0)
    else:
        raise ValueError(f"unknown sampling method: '{method}'")

# }}}


# {{{ generation

def make_gmsh_sphere(
        order: int, cls: type, *,
        radius: float = 1.0,
        length: float = 0.3) -> Mesh:
    from meshmode.mesh.io import ScriptSource
    from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup
    if issubclass(cls, SimplexElementGroup):
        script = ScriptSource(
            """
            Mesh.CharacteristicLengthMax = %(length)g;
            Mesh.HighOrderOptimize = 1;
            Mesh.Algorithm = 1;

            SetFactory("OpenCASCADE");
            Sphere(1) = {0, 0, 0, %(radius)g};
            """ % {"radius": radius, "length": length},
            "geo")
    elif issubclass(cls, TensorProductElementGroup):
        script = ScriptSource(
            """
            Mesh.CharacteristicLengthMax = %(length)g;
            Mesh.HighOrderOptimize = 1;
            Mesh.Algorithm = 6;

            SetFactory("OpenCASCADE");
            Sphere(1) = {0, 0, 0, %(radius)g};
            Recombine Surface "*" = 0.0001;
            """ % {"radius": radius, "length": length},
            "geo")
    else:
        raise TypeError

    from meshmode.mesh.io import generate_gmsh
    return generate_gmsh(
            script,
            order=order,
            dimensions=2,
            force_ambient_dim=3,
            target_unit="MM",
            )

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


# {{{ get_discr_nodes

@memoize_on_first_arg
def get_discr_nodes(discr: Discretization) -> np.ndarray:
    from arraycontext import thaw, flatten
    actx = discr._setup_actx        # pylint: disable=protected-access

    return actx.to_numpy(
            flatten(thaw(discr.nodes(), actx), actx),
            ).reshape(discr.ambient_dim, -1)

# }}}


# {{{ get_point_radius_and_center

def get_point_radius_and_center(points: np.ndarray) -> Tuple[float, np.ndarray]:
    center = np.mean(points, axis=1)
    radius = np.max(la.norm(points - center.reshape(-1, 1), ord=2, axis=0))

    return radius, center

# }}}


# {{{ find_farthest_apart_cluster

def find_farthest_apart_node(nodes: np.ndarray, center: np.ndarray) -> np.ndarray:
    dists = la.norm(nodes - center.reshape(-1, 1), axis=0)
    return np.argmax(dists)


def find_nodes_around_center(
        nodes: np.ndarray, center: np.ndarray, nnodes: int) -> np.ndarray:
    dists = la.norm(nodes - center.reshape(-1, 1), axis=0)

    radius = 1.0e-2
    while True:
        indices = np.where(dists < radius)[0]
        if indices.size >= nnodes:
            break

        radius = 1.25 * radius

    return indices


def find_farthest_apart_cluster(
        actx: ArrayContext,
        discr: Discretization,
        cindex: IndexList,
        itarget: Optional[int] = None,
        target_center: Optional[np.ndarray] = None) -> int:
    nodes = get_discr_nodes(discr)
    if target_center is None:
        target_nodes = cindex.cluster_take(nodes.T, itarget).T
        target_center = np.mean(target_nodes, axis=1)

    max_index = None
    max_dists = -np.inf

    for jsource in range(cindex.nclusters):
        if itarget == jsource:
            continue

        source_nodes = cindex.cluster_take(nodes.T, jsource).T
        source_center = np.mean(source_nodes, axis=1)

        dist = la.norm(target_center - source_center)
        if dist > max_dists:
            max_dists = dist
            max_index = jsource

    return max_index

# }}}
