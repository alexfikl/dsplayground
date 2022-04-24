from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.linalg as la

import scipy.linalg.interpolative as sli    # pylint: disable=no-name-in-module

import matplotlib.pyplot as mp
import matplotlib.patches as patch

import logging
logger = logging.getLogger(__name__)


# {{{ skeletonize

def make_p2p_skeletonization_wrangler(places, sym_op, sym_sigma, *, auto_where):
    from pytential.symbolic.matrix import P2PClusterMatrixBuilder
    from pytential.linalg.skeletonization import make_skeletonization_wrangler
    return make_skeletonization_wrangler(
            places, sym_op, sym_sigma,
            auto_where=auto_where,
            _weighted_proxy=False,
            _proxy_source_cluster_builder=P2PClusterMatrixBuilder,
            _proxy_target_cluster_builder=P2PClusterMatrixBuilder,
            _neighbor_cluster_builder=P2PClusterMatrixBuilder,
            )


def make_qbx_skeletonization_wrangler(places, sym_op, sym_sigma, *, auto_where):
    from pytential.linalg.skeletonization import make_skeletonization_wrangler
    return make_skeletonization_wrangler(
            places, sym_op, sym_sigma,
            auto_where=auto_where)

# }}}


# {{{ run

@dataclass(frozen=True, unsafe_hash=True)
class Geometry:
    # geometry
    target_order: int = 4
    source_ovsmp: int = 4
    qbx_order: int = 4
    group_cls_name = "simplex"

    # skeletonization
    id_eps: float = 1.0e-15
    nclusters: int = 6
    proxy_approx_count: int = 32
    proxy_radius_factor: float = 1.0
    tree_kind = "adaptive-level-restricted"

    mode: str = "qbx"


@dataclass(frozen=True, unsafe_hash=True)
class CurveGeometry(Geometry):
    radius: float = 3.0
    resolution: int = 256

    def get_mesh(self):
        import meshmode.mesh.generation as mgen
        return mgen.make_curve_mesh(
                lambda t: self.radius * mgen.starfish(t),
                np.linspace(0.0, 1.0, self.resolution + 1),
                self.target_order, closed=True,
                )


@dataclass(frozen=True, unsafe_hash=True)
class SurfaceGeometry(Geometry):
    radius: float = 1.0
    resolution: float = 0.4

    def get_mesh(self):
        import meshmode.mesh as mmesh
        if self.group_cls_name == "simplex":
            group_cls = mmesh.SimplexElementGroup
        elif self.group_cls_name == "tensor":
            group_cls = mmesh.TensorProductElementGroup
        else:
            raise ValueError(
                    f"unknown mesh element group class: '{self.group_cls_name}'")

        import dsplayground as ds
        return ds.make_gmsh_sphere(self.target_order, group_cls,
                radius=self.radius,
                length=self.resolution)


def run_error_model(ctx_factory, *,
        ambient_dim: int = 2, visualize: bool = True,
        **kwargs: Any) -> None:
    import dsplayground as ds
    actx = ds.get_cl_array_context(ctx_factory)

    np.random.seed(42)
    sli.seed(42)

    if ambient_dim == 2:
        case = CurveGeometry(**kwargs)
    elif ambient_dim == 3:
        case = SurfaceGeometry(**kwargs)
    else:
        raise ValueError(f"unsupported dimension: {ambient_dim}")

    basename = f"factorization_rank_{case.mode}_{ds.dc_hash(case)}"

    # {{{ geometry

    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureGroupFactory
    from meshmode.discretization import Discretization
    mesh = case.get_mesh()
    group_factory = InterpolatoryQuadratureGroupFactory(case.target_order)
    pre_density_discr = Discretization(actx, mesh, group_factory)

    from pytential.qbx import QBXLayerPotentialSource
    qbx = QBXLayerPotentialSource(pre_density_discr,
            fine_order=case.source_ovsmp * case.target_order,
            qbx_order=case.qbx_order,
            fmm_order=False, fmm_backend=None,
            _disable_refinement=True)

    from pytential import GeometryCollection
    places = GeometryCollection(qbx, auto_where="qbx")
    dofdesc = places.auto_source.to_stage2()

    density_discr = places.get_discretization(dofdesc.geometry, dofdesc.discr_stage)

    logger.info("nelements:     %d", density_discr.mesh.nelements)
    logger.info("ndofs:         %d", density_discr.ndofs)

    # }}}

    # {{{ wrangler

    from pytential.linalg import partition_by_nodes
    max_particles_in_box = density_discr.ndofs // case.nclusters
    cindex = partition_by_nodes(actx, places,
            dofdesc=dofdesc,
            tree_kind=case.tree_kind,
            max_particles_in_box=max_particles_in_box)

    from pytential.linalg import TargetAndSourceClusterList
    tgt_src_index = TargetAndSourceClusterList(cindex, cindex)

    from sumpy.kernel import LaplaceKernel
    kernel = LaplaceKernel(ambient_dim)

    from pytential import sym
    sym_sigma = sym.var("sigma")
    sym_op = sym.S(kernel, sym_sigma, qbx_forced_limit=+1)

    from pytential.symbolic.execution import _prepare_expr
    sym_op_prep = _prepare_expr(places, sym_op)

    if case.mode == "p2p":
        from pytential.symbolic.matrix import P2PMatrixBuilder as MatrixBuilder
        wrangler = make_p2p_skeletonization_wrangler(
                places, sym_op, sym_sigma,
                auto_where=dofdesc)
    elif case.mode == "qbx":
        from pytential.symbolic.matrix import MatrixBuilder
        wrangler = make_qbx_skeletonization_wrangler(
                places, sym_op, sym_sigma,
                auto_where=dofdesc)
    else:
        raise ValueError(f"unknown mode: '{case.mode}'")

    mat = MatrixBuilder(
        actx,
        dep_expr=sym_sigma,
        other_dep_exprs=[],
        dep_source=qbx,
        dep_discr=density_discr,
        places=places,
        context={},
        _weighted=wrangler.weighted_sources)(sym_op_prep)

    # }}}

    # {{{ clusters

    from pytential.linalg import QBXProxyGenerator
    proxy_generator = QBXProxyGenerator(places,
            radius_factor=case.proxy_radius_factor,
            approx_nproxy=8)
    pxy = proxy_generator(
            actx, dofdesc, tgt_src_index.sources
            ).to_numpy(actx, stack=True)

    # find biggest cluster and use it as a target
    n_cluster_sizes = np.diff(cindex.starts)
    itarget = np.argmax(n_cluster_sizes)

    # find source clusters: nearest
    target_center = pxy.centers[:, itarget]
    dists = la.norm(
            pxy.centers
            - target_center.reshape(-1, 1), axis=0)

    dists[itarget] = -np.inf
    jsource_far = np.argmax(dists)
    dists[itarget] = np.inf
    jsource_near = np.argmin(dists)

    logger.info("itarget %2d jsource far %2d near %2d",
            itarget, jsource_far, jsource_near)
    assert itarget != jsource_far and itarget != jsource_near

    if visualize:
        plot_skeletonization_geometry(actx, pxy, basename,
                itarget=itarget, jsource_far=jsource_far, jsource_near=jsource_near,
                )

    # }}}

    # {{{ compute errors

    from pytential.linalg.skeletonization import \
            _skeletonize_block_by_proxy_with_mats

    nproxies = np.linspace(4, n_cluster_sizes[itarget], 8, dtype=np.int64)
    skel_rank = np.empty(nproxies.shape, dtype=np.int64)
    error_far = np.empty(nproxies.shape, dtype=np.float64)
    error_near = np.empty(nproxies.shape, dtype=np.float64)

    for i, nproxy in enumerate(nproxies):
        proxy_generator = QBXProxyGenerator(places,
                radius_factor=case.proxy_radius_factor,
                approx_nproxy=nproxy)

        L, R, skel_tgt_src_index, _, _ = (                      # noqa: N806
                _skeletonize_block_by_proxy_with_mats(
                    actx, 0, 0, places, proxy_generator, wrangler, tgt_src_index,
                    id_eps=case.id_eps,
                    max_particles_in_box=max_particles_in_box))

        from pytential.linalg import SkeletonizationResult
        skeleton = SkeletonizationResult(
                L=L, R=R,
                tgt_src_index=tgt_src_index,
                skel_tgt_src_index=skel_tgt_src_index)

        # skeletonization rank
        skel_rank[i] = L[itarget, itarget].shape[1]

        from pytential.linalg.utils import cluster_skeletonization_error
        blk_err_l, blk_err_r = cluster_skeletonization_error(
                mat, skeleton, ord=2, relative=True)

        error_far[i] = blk_err_l[itarget, jsource_far]
        error_near[i] = blk_err_l[itarget, jsource_near]

        logger.info("nproxy %4d rank %4d error far %.12e near %.12e",
                nproxy, skel_rank[i], error_far[i], error_near[i])

    filename = f"{basename}.npz"
    np.savez_compressed(filename,
            nproxies=nproxies,
            skel_rank=skel_rank,
            max_rank=n_cluster_sizes[itarget],
            error_far=error_far,
            error_near=error_near,
            )

    # }}}

    if visualize:
        plot_error_model(filename)

# }}}


# {{{ plot


def plot_skeletonization_geometry(actx, pxy, basename, *,
        itarget, jsource_far, jsource_near):
    srcindex = pxy.srcindex
    dofdesc = pxy.dofdesc
    places = pxy.places
    if places.ambient_dim != 2:
        return

    import dsplayground as ds
    density_discr = places.get_discretization(dofdesc.geometry, dofdesc.discr_stage)
    nodes = ds.get_discr_nodes(density_discr)

    fig = mp.figure()
    ax = fig.gca()

    ax.plot(pxy.centers[0], pxy.centers[1], "ko")

    for i in range(srcindex.nclusters):
        if i == itarget:
            label = "target"
        elif i == jsource_far:
            label = "far"
        elif i == jsource_near:
            label = "near"
        else:
            label = None

        isrc = srcindex.cluster_indices(i)
        line, = ax.plot(nodes[0][isrc], nodes[1][isrc], "o", ms=5, label=label)
        if label is not None:
            color = line.get_color()
        else:
            color = "black"

        c = patch.Circle(pxy.centers[:, i], pxy.radii[i], color=color, alpha=0.2)
        ax.add_artist(c)
        ax.text(*pxy.centers[:, i], f"{i}",
                fontweight="bold", color="white", ha="center", va="center")

    ax.legend()
    ax.set_aspect("equal")
    fig.savefig(f"{basename}_geometry")


def plot_error_model(filename: str):
    import pathlib
    filename = pathlib.Path(filename)
    basename = filename.with_suffix("")

    d = np.load(filename)

    import dsplayground as ds
    fig = mp.figure()

    outfile = basename.with_stem(f"{basename.stem}_rank")
    with ds.axis(fig, outfile) as ax:
        ax.plot(d["nproxies"], d["skel_rank"])
        ax.axhline(d["max_rank"], color="k", ls="--")

        ax.set_xlabel("$Proxy$")
        ax.set_ylabel("$Rank$")

# }}}


if __name__ == "__main__":
    import sys
    import pyopencl as cl
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        run_error_model(cl.create_some_context)
