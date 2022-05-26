from dataclasses import dataclass
from typing import Any, Optional

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


def get_mat_no_nbr(self, i):
    shape = self.pxyindex.cluster_shape(i, i)
    pxymat_i = self.pxyindex.flat_cluster_take(self.pxymat, i).reshape(*shape)

    return [pxymat_i]


def far_cluster_target_skeletonization_error(
        mat, skeleton, *, i, j, ord=2, relative=True
        ) -> float:
    assert i != j

    # skeleton matrix indices
    s_tgt = np.searchsorted(
        skeleton.tgt_src_index.targets.cluster_indices(i),
        skeleton.skel_tgt_src_index.targets.cluster_indices(i),
        )
    sblk = mat[s_tgt, :]

    # compute cluster errors
    error = la.norm(mat - skeleton.L[i, i] @ sblk)
    if relative:
        error = error / la.norm(mat)

    return error


def near_cluster_target_skeletonization_error(
        pxy, mat, skeleton, *, i, j, ord=2, relative=True,
        ) -> float:
    assert i != j

    import dsplayground as ds
    discr = pxy.discr
    nodes = ds.get_discr_nodes(discr)

    # find source indices outside of proxy ball only
    src_nodes = skeleton.tgt_src_index.sources.cluster_take(nodes.T, j).T
    mask = la.norm(
            src_nodes - pxy.centers[:, i].reshape(-1, 1), axis=0
            ) > pxy.radii[i]

    # full matrix indices
    blk = mat[:, mask]

    # skeleton matrix indices
    s_tgt = np.searchsorted(
        skeleton.tgt_src_index.targets.cluster_indices(i),
        skeleton.skel_tgt_src_index.targets.cluster_indices(i),
        )
    sblk = mat[:, mask][s_tgt, :]

    # compute cluster errors
    error = la.norm(blk - skeleton.L[i, i] @ sblk)
    if relative:
        error = error / la.norm(blk)

    return error

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
    id_eps: float = 1.0e-5
    nclusters: int = 6
    tree_kind = "adaptive-level-restricted"

    expn: str = "qbx"
    mode: str = "full"


@dataclass(frozen=True, unsafe_hash=True)
class CurveGeometry(Geometry):
    radius: float = 3.0
    resolution: int = 512

    def get_mesh(self):
        import meshmode.mesh.generation as mgen
        return mgen.make_curve_mesh(
                lambda t: self.radius * mgen.starfish(t),
                np.linspace(0.0, 1.0, self.resolution + 1),
                self.target_order, closed=True,
                )


@dataclass(frozen=True, unsafe_hash=True)
class SurfaceGeometry(Geometry):
    nclusters: int = 12

    radius: float = 1.0
    resolution: float = 0.1

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

    basename = f"factorization_rank_{case.expn}_{case.mode}_{ds.dc_hash(case)}"

    if case.mode == "full":
        pass
    elif case.mode == "nnbr":
        # NOTE: ugly monkeypatching so that we don't actually skeletonize the
        # nearfield neighbor nodes
        from pytential.linalg import skeletonization
        skeletonization._ProxyNeighborEvaluationResult.__getitem__ = get_mat_no_nbr
    else:
        raise ValueError(f"unknown mode: '{case.mode}'")

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
    logger.info("%s", np.diff(cindex.starts))

    from pytential.linalg import TargetAndSourceClusterList
    tgt_src_index = TargetAndSourceClusterList(cindex, cindex)

    from sumpy.kernel import LaplaceKernel
    kernel = LaplaceKernel(ambient_dim)

    from pytential import sym
    sym_sigma = sym.var("sigma")
    sym_op = sym.S(kernel, sym_sigma, qbx_forced_limit=+1)

    if case.expn == "p2p":
        wrangler = make_p2p_skeletonization_wrangler(
                places, sym_op, sym_sigma,
                auto_where=dofdesc)
    elif case.expn == "qbx":
        wrangler = make_qbx_skeletonization_wrangler(
                places, sym_op, sym_sigma,
                auto_where=dofdesc)
    else:
        raise ValueError(f"unknown expansion: '{case.expn}'")

    # }}}

    # {{{ clusters

    from pytential.linalg import QBXProxyGenerator
    # NOTE: this proxy generator is only constructed to get the centers
    proxy_generator = QBXProxyGenerator(places, radius_factor=1.0, approx_nproxy=8)
    pxy = proxy_generator(
            actx, dofdesc, tgt_src_index.sources, include_cluster_radii=True,
            ).to_numpy(actx, stack_nodes=True)

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
        if ambient_dim == 2:
            plot_skeletonization_geometry = plot_skeletonization_curve_geometry
        else:
            plot_skeletonization_geometry = plot_skeletonization_surface_geometry

        plot_skeletonization_geometry(
                actx, pxy, basename,
                itarget=itarget,
                jsource_far=jsource_far,
                jsource_near=jsource_near,
                )

    # furthest target point
    nodes = ds.get_discr_nodes(density_discr)
    tgt_nodes = (
            tgt_src_index.targets.cluster_take(nodes.T, itarget).T
            - target_center.reshape(-1, 1)) / pxy.radii[itarget]

    itarget_max = np.argmax(la.norm(tgt_nodes, axis=0))
    logger.info("target %d point (%.5e, %.5e) distance %.5e",
            itarget_max, *tgt_nodes[:, itarget_max],
            la.norm(tgt_nodes[:, itarget_max]))

    if visualize:
        plot_complex_kernel(basename,
                z0=tgt_nodes[0, itarget_max] + 1j * tgt_nodes[1, itarget_max])

    1/0

    # }}}

    # {{{ dense matrix

    from pytools.obj_array import make_obj_array
    from pytential.linalg.utils import make_index_list
    tgt_index = make_index_list(make_obj_array([
        cindex.cluster_indices(itarget),
        cindex.cluster_indices(itarget),
        ]))
    src_index = make_index_list(make_obj_array([
        cindex.cluster_indices(jsource_far),
        cindex.cluster_indices(jsource_near),
        ]))
    dense_tgt_src_index = TargetAndSourceClusterList(tgt_index, src_index)

    from pytential.linalg.utils import make_flat_cluster_diag
    mat = wrangler._evaluate_expr(
            actx, places, wrangler.neighbor_cluster_builder, dense_tgt_src_index,
            wrangler.exprs[0], 0, _weighted=wrangler.weighted_sources)
    mat = make_flat_cluster_diag(mat, dense_tgt_src_index)

    # }}}

    # {{{ compute errors

    from pytential.linalg.skeletonization import \
            _skeletonize_block_by_proxy_with_mats

    factors = np.linspace(1.0, 1.5, 4, endpoint=True)
    id_tolerances = 10.0 ** -np.linspace(2, 15, 16)
    nproxies = np.logspace(
            np.log10(4),
            np.log10(n_cluster_sizes[itarget] + 128),
            12, dtype=np.int64)
    shape = (factors.size, nproxies.size, id_tolerances.size)

    skel_id_rank = np.empty(shape, dtype=np.int64)
    skel_num_rank = np.empty(shape, dtype=np.int64)
    error_far = np.empty(shape, dtype=np.float64)
    error_near = np.empty(shape, dtype=np.float64)

    from itertools import product
    for i, proxy_radius_factor in enumerate(factors):
        for (j, nproxy), (k, id_eps) in product(
                enumerate(nproxies), enumerate(id_tolerances)):
            proxy_generator = QBXProxyGenerator(places,
                    radius_factor=proxy_radius_factor,
                    approx_nproxy=nproxy)

            L, R, skel_tgt_src_index, _, tgt = (                        # noqa: N806
                    _skeletonize_block_by_proxy_with_mats(
                        actx, 0, 0, places, proxy_generator, wrangler, tgt_src_index,
                        id_eps=id_eps,
                        max_particles_in_box=max_particles_in_box))
            tgt_mat = np.hstack(tgt[itarget])

            from pytential.linalg import SkeletonizationResult
            skeleton = SkeletonizationResult(
                    L=L, R=R,
                    tgt_src_index=tgt_src_index,
                    skel_tgt_src_index=skel_tgt_src_index)

            # skeletonization rank
            skel_id_rank[i, j, k] = L[itarget, itarget].shape[1]
            skel_num_rank[i, j, k] = la.matrix_rank(tgt_mat, tol=id_eps)

            error_far[i, j, k] = far_cluster_target_skeletonization_error(
                    mat[0, 0], skeleton, i=itarget, j=jsource_far)
            error_near[i, j, k] = near_cluster_target_skeletonization_error(
                    tgt.pxy.to_numpy(actx, stack_nodes=True),
                    mat[1, 1], skeleton, i=itarget, j=jsource_near)

            logger.info(
                    "shape %4dx%4d nproxy %4d rank %4d / %4d "
                    + "id_eps %.5e far %.12e near %.12e",
                    tgt_mat.shape[0], tgt_mat.shape[1],
                    nproxy, skel_id_rank[i, j, k], skel_num_rank[i, j, k],
                    id_eps, error_far[i, j, k], error_near[i, j, k])

        logger.info("")

    filename = f"{basename}.npz"
    np.savez_compressed(filename,
            parameters=ds.dc_dict(case),
            # geometry
            ambient_dim=ambient_dim,
            # skeletonization info
            id_eps=case.id_eps,
            itarget=itarget, jsource_far=jsource_far, jsource_near=jsource_near,
            n_cluster_sizes=n_cluster_sizes,
            proxy_radii=pxy.radii,
            proxy_centers=pxy.centers,
            cluster_radii=pxy._cluster_radii,
            # skeletonization results
            proxy_factors=factors,
            id_tolerances=id_tolerances,
            nproxies=nproxies,
            skel_id_rank=skel_id_rank,
            skel_num_rank=skel_num_rank,
            error_far=error_far,
            error_near=error_near,
            )

    # }}}

    if visualize:
        _plot_error_model(filename)

# }}}


# {{{ plot


def plot_skeletonization_curve_geometry(actx, pxy, basename, *,
        itarget, jsource_far, jsource_near):
    srcindex = pxy.srcindex
    dofdesc = pxy.dofdesc
    places = pxy.places
    assert places.ambient_dim == 2

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


def plot_skeletonization_surface_geometry(actx, pxy, basename, *,
        itarget, jsource_far, jsource_near):
    dofdesc = pxy.dofdesc
    places = pxy.places
    assert places.ambient_dim == 3

    density_discr = places.get_discretization(dofdesc.geometry, dofdesc.discr_stage)
    target_order = min(grp.order for grp in density_discr.groups)

    from meshmode.mesh.generation import generate_sphere
    ref_mesh = generate_sphere(1.0, target_order, uniform_refinement_rounds=1)

    from meshmode.mesh.processing import affine_map
    meshes = [density_discr.mesh]
    indices = [0, itarget, jsource_near, jsource_far]
    for i in indices[1:]:
        meshes.append(
                affine_map(ref_mesh, A=pxy.radii[i], b=pxy.centers[:, i])
                )

    from meshmode.mesh.processing import merge_disjoint_meshes
    mesh = merge_disjoint_meshes(meshes)

    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureGroupFactory
    from meshmode.discretization import Discretization
    discr = Discretization(actx, mesh,
            InterpolatoryQuadratureGroupFactory(target_order))

    marker = discr.zeros(actx)
    marker = type(marker)(actx, tuple([
        indices[i] + subary for i, subary in enumerate(marker)
        ]))

    from meshmode.discretization.visualization import make_visualizer
    vis = make_visualizer(actx, discr, target_order)
    vis.write_vtk_file(f"{basename}_geometry.vtu", [
        ("marker", marker)
        ], overwrite=True)


def plot_complex_kernel(basename, *, z0):
    def f(z):
        return np.log(z0 - np.exp(1j * z))

    import dsplayground as ds
    fig = mp.figure()

    import cplot
    outfile = f"{basename}_complex"
    with ds.axis(fig, outfile) as _:
        cplot.plot(f,
                (0.0, 2.0 * np.pi, 1024),
                (-np.pi, np.pi, 1024))

    outfile = f"{basename}_singularity"
    with ds.axis(fig, outfile) as ax:
        x = np.linspace(-1.0, 1.0, 512)
        x, y = np.meshgrid(x, x)
        mask = np.sqrt(x**2 + y**2) < 1.0

        z = np.imag(-1j * np.log(x + 1j * y))
        z[~mask] = np.nan

        im = ax.contourf(x, y, z, levels=32)
        ax.contour(x, y, z, colors="k", linewidths=(1,), levels=32)
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, shrink=0.75)


def plot_error_model(glob: Optional[str]) -> None:
    import pathlib
    for filename in pathlib.Path().glob(glob):
        logger.info("%s", filename)
        _plot_error_model(filename)


def _plot_error_model(filename: str) -> None:
    import pathlib
    filename = pathlib.Path(filename)
    basename = filename.with_suffix("")
    title = basename.name.split("_")[2].upper()

    d = np.load(filename)
    # ambient_dim = d["ambient_dim"]
    id_eps = d["id_eps"]

    factors = d["proxy_factors"]
    id_tolerances = d["id_tolerances"]
    nproxies = d["nproxies"]

    centers = d["proxy_centers"]
    cradii = d["cluster_radii"]

    itarget = d["itarget"]
    jsource = d["jsource_far"]

    source_radius_far = (
            la.norm(centers[:, itarget] - centers[:, jsource])
            - cradii[jsource])
    rho = cradii[itarget] / source_radius_far

    logger.info("rho %.15f id_eps %.5e", rho, id_eps)

    import dsplayground as ds
    fig = mp.figure()
    m_id_tolerances, m_nproxies = np.meshgrid(np.log10(id_tolerances), nproxies)

    # {{{ decay coefficient

    def _plot_decay(_ranks, _errors, suffix):
        outfile = basename.with_stem(f"{basename.stem}_decay_{suffix}")
        with ds.axis(fig, outfile) as ax:
            a = np.empty(id_tolerances.shape)

            # NOTE: after 8, the errors mostly bottom out
            maxj = 6

            for i in range(factors.size):
                for k in range(a.size):
                    rank = _ranks[i, :maxj, k]
                    error = _errors[i, :maxj, k]

                    coeffs = np.polyfit(-rank - 1, np.log(error), 1)
                    a[k] = coeffs[-2]

                ax.semilogx(id_tolerances, a, label=f"{factors[i]:.2f}")

            ax.set_xlabel(r"$\epsilon_{ID}$")
            ax.set_ylabel("$a$")
            ax.set_ylim([0.0, 1.6])

            ax.legend(
                    title=title,
                    bbox_to_anchor=(1.02, 0, 0.25, 1.0),
                    loc="upper left", mode="expand",
                    borderaxespad=0, ncol=1)

    _plot_decay(d["skel_id_rank"], d["error_far"], "far")
    _plot_decay(d["skel_id_rank"], d["error_near"], "near")

    # }}}

    # {{{ plot rank

    def _plot_rank_contour(_ranks):
        for i in range(factors.size):
            outfile = basename.with_stem(f"{basename.stem}_rank_{i:02d}")

            with ds.axis(fig, outfile) as ax:
                im = ax.contourf(m_nproxies, m_id_tolerances, _ranks[i],
                        levels=32)
                ax.contour(m_nproxies, m_id_tolerances, _ranks[i],
                        colors="k", linewidths=(1,), levels=32)
                fig.colorbar(im, ax=ax)

                ax.set_xlabel("$Proxy$")
                ax.set_ylabel(r"$\epsilon_{ID}$")

    # _plot_rank_contour(d["skel_id_rank"])

    # }}}

    # {{{ plot errors

    def _plot_error(_errors, k, suffix):
        outfile = basename.with_stem(f"{basename.stem}_error_{suffix}")
        with ds.axis(fig, outfile) as ax:
            for i in range(factors.size):
                ax.semilogy(nproxies, _errors[i, :, k], "o-",
                        label=f"{factors[i]:.2f}")

            ax.axhline(id_tolerances[k], color="k", ls="--")
            ax.set_xlabel(r"$Proxy$")
            ax.set_ylabel("$Relative~ Error$")
            ax.set_ylim([1.0e-16, 1.0])

            ax.legend(
                    title=title,
                    bbox_to_anchor=(1.02, 0, 0.25, 1.0),
                    loc="upper left", mode="expand",
                    borderaxespad=0, ncol=1)

    _plot_error(d["error_far"], 3, "far")
    _plot_error(d["error_near"], 3, "near")

    def _plot_error_contour(_errors, suffix):
        for i in range(factors.size):
            outfile = basename.with_stem(f"{basename.stem}_error_{suffix}_{i:02d}")
            with ds.axis(fig, outfile) as ax:
                log_error = np.log10(_errors[i])

                im = ax.contourf(m_nproxies, m_id_tolerances, log_error,
                        levels=32)
                ax.contour(m_nproxies, m_id_tolerances, log_error,
                        colors="k", linewidths=(1,), levels=32)
                fig.colorbar(im, ax=ax)

                ax.set_xlabel("$Proxy$")
                ax.set_ylabel(r"$\epsilon_{ID}$")

    # _plot_error_contour(d["error_far"], "far")
    # _plot_error_contour(d["error_near"], "near")

    # }}}

# }}}


if __name__ == "__main__":
    import sys
    import pyopencl as cl
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        run_error_model(cl.create_some_context)
