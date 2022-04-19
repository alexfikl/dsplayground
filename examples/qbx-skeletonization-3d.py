from dataclasses import dataclass
from functools import partial
from typing import Any, Optional

import numpy as np
import numpy.linalg as la

import scipy.linalg.interpolative as sli    # pylint: disable=no-name-in-module

import matplotlib.pyplot as mp

import logging
logger = logging.getLogger(__name__)


# {{{ skeletonize and compute errors

@dataclass(frozen=True)
class ErrorInfo:
    error: float
    id_rank: int
    proxy_rank: int
    interaction_mat: np.ndarray
    rec_mat: np.ndarray
    error_mat: np.ndarray


def rnorm(x, y, ord=None):          # pylint: disable=redefined-builtin
    ynorm = la.norm(y, ord=ord)
    if ynorm < 1.0e-15:
        ynorm = 1

    return la.norm(x - y, ord=ord) / ynorm


def qbx_interaction_mat(
        actx, places, *,
        qbx_order: int,
        source: str, target: str,
        ):
    source_discr = places.get_discretization(source)
    target_discr = places.get_discretization(target)

    from sumpy.kernel import LaplaceKernel
    kernel = LaplaceKernel(places.ambient_dim)

    from sumpy.expansion.local import LineTaylorLocalExpansion
    expansion = LineTaylorLocalExpansion(kernel, qbx_order)

    from sumpy.qbx import LayerPotentialMatrixGenerator
    gen = LayerPotentialMatrixGenerator(actx.context,
                                        expansion=expansion,
                                        source_kernels=(kernel,),
                                        target_kernels=(kernel,))

    _, (mat,) = gen(actx.queue,
                    targets=target_discr.nodes(),
                    sources=source_discr.nodes(),
                    centers=target_discr.expansion_centers(+1),
                    expansion_radii=target_discr.expansion_radii())

    mat = actx.to_numpy(mat)

    return mat


def compute_target_reconstruction_error(
        actx, wrangler, places, mat_indices, pxy, *,
        source_dd, target_dd, proxy_dd, id_eps,
        ord=2, verbose=True):           # pylint: disable=redefined-builtin
    # {{{ evaluate matrices

    from pytools import memoize_in

    @memoize_in(mat_indices, (compute_target_reconstruction_error, "interaction"))
    def _interaction_mat():
        mat = wrangler._evaluate_expr(      # pylint: disable=protected-access
            actx, places,
            wrangler.neighbor_cluster_builder,
            mat_indices,
            wrangler.exprs[0],
            idomain=0,
            _weighted=wrangler.weighted_targets,
            )
        return mat.reshape(mat_indices.cluster_shape(0, 0))

    proxy_mat, pxyindices = wrangler.evaluate_target_farfield(
            actx, places, pxy, None,
            ibrow=0, ibcol=0)
    proxy_mat = proxy_mat.reshape(pxyindices.cluster_shape(0, 0))
    interaction_mat = _interaction_mat()

    logger.info("%.12e %.12e", la.norm(interaction_mat), la.norm(proxy_mat))

    # }}}

    # {{{ skeletonize

    k, idx, proj = sli.interp_decomp(proxy_mat.T, id_eps)
    P = sli.reconstruct_interp_matrix(idx, proj).T      # noqa: N806
    idx = idx[:k]

    id_error = rnorm(proxy_mat, P @ proxy_mat[idx, :], ord=ord)
    proxy_rank = la.matrix_rank(proxy_mat, tol=id_eps)

    if verbose:
        logger.info("id_rank:   %3d num_rank %3d nproxy %4d",
                idx.size, proxy_rank, proxy_mat.shape[1])
        logger.info("id_error:  %.15e (eps %.5e)", id_error, id_eps)
        logger.info("\n")

    # }}}

    # {{{ compute reconstruction error

    rec_mat = P @ interaction_mat[idx, :]
    error_mat = interaction_mat - rec_mat
    rec_error = rnorm(rec_mat, interaction_mat, ord=ord)

    # }}}

    return ErrorInfo(
            error=rec_error, id_rank=k, proxy_rank=proxy_rank,
            interaction_mat=interaction_mat,
            rec_mat=rec_mat,
            error_mat=error_mat)

# }}}


# {{{ set up

def make_gmsh_sphere(order: int, cls: type):
    from meshmode.mesh.io import ScriptSource
    from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup
    if issubclass(cls, SimplexElementGroup):
        script = ScriptSource(
            """
            Mesh.CharacteristicLengthMax = 0.4;
            Mesh.HighOrderOptimize = 1;
            Mesh.Algorithm = 1;

            SetFactory("OpenCASCADE");
            Sphere(1) = {0, 0, 0, 1.5};
            """,
            "geo")
    elif issubclass(cls, TensorProductElementGroup):
        script = ScriptSource(
            """
            Mesh.CharacteristicLengthMax = 0.1;
            Mesh.HighOrderOptimize = 1;
            Mesh.Algorithm = 6;

            SetFactory("OpenCASCADE");
            Sphere(1) = {0, 0, 0, 0.5};
            Recombine Surface "*" = 0.0001;
            """,
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


def make_geometry_collection(
        actx, *,
        nelements: int,
        target_order: int,
        source_ovsmp: int,
        qbx_order: int,
        source_name: str = "qbx",
        mesh_name: str = "gmsh_sphere",
        ):
    import meshmode.mesh as mmesh
    if mesh_name == "torus":
        import meshmode.mesh.generation as mgen
        mesh = mgen.generate_torus(1, 0.5,
                n_major=nelements, n_minor=nelements // 2,
                order=target_order,
                # group_cls=mmesh.SimplexElementGroup,
                group_cls=mmesh.TensorProductElementGroup,
                )
    elif mesh_name == "gmsh_sphere":
        mesh = make_gmsh_sphere(target_order, mmesh.SimplexElementGroup)
    else:
        raise ValueError(f"unknown mesh: '{mesh_name}'")

    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureGroupFactory
    from meshmode.discretization import Discretization
    pre_density_discr = Discretization(actx, mesh,
            group_factory=InterpolatoryQuadratureGroupFactory(target_order))

    from pytential.qbx import QBXLayerPotentialSource
    qbx = QBXLayerPotentialSource(pre_density_discr,
            fine_order=source_ovsmp * target_order,
            qbx_order=qbx_order,
            fmm_order=False, fmm_backend=None,
            _disable_refinement=True)

    from pytential import GeometryCollection
    places = GeometryCollection(qbx, auto_where=source_name)

    return places


def make_cluster_indices(
        actx, density_discr, *,
        nclusters: int,
        itarget: Optional[int], jsource: Optional[int],
        basename: Optional[str] = None, visualize: bool = False,
        ):
    from pytential.linalg.proxy import partition_by_nodes
    source_partition = partition_by_nodes(actx, density_discr,
            max_particles_in_box=density_discr.ndofs // (64),
            tree_kind="adaptive-level-restricted")
    target_partition = partition_by_nodes(actx, density_discr,
            max_particles_in_box=density_discr.ndofs // (48),
            tree_kind="adaptive-level-restricted")

    logger.info("nclusters %5d got source %5d target %5d", nclusters,
                source_partition.nclusters, target_partition.nclusters)

    import dsplayground as ds

    def get_center(partition, idx):
        nodes = ds.get_discr_nodes(density_discr)
        nodes = partition.cluster_take(nodes.T, idx).T
        return np.mean(nodes, axis=1)

    if itarget is None and jsource is None:
        itarget = 0
        jsource = ds.find_farthest_apart_cluster(
            actx, density_discr, source_partition,
            target_center=get_center(target_partition, itarget))
    elif itarget is None and jsource is not None:
        itarget = ds.find_farthest_apart_cluster(
            actx, density_discr, target_partition,
            target_center=get_center(source_partition, jsource))
    elif itarget is not None and jsource is None:
        jsource = ds.find_farthest_apart_cluster(
            actx, density_discr, source_partition,
            target_center=get_center(target_partition, itarget))
    else:
        assert itarget is not None and 0 <= itarget < target_partition.nclusters
        assert jsource is not None and 0 <= jsource < source_partition.nclusters

    if visualize:
        target_cluster_sizes = np.array([
            target_partition.cluster_size(i) for i in range(target_partition.nclusters)
        ])
        logger.info("block sizes: %s", list(target_cluster_sizes))
        source_cluster_sizes = np.array([
            source_partition.cluster_size(i) for i in range(source_partition.nclusters)
        ])
        logger.info("block sizes: %s", list(source_cluster_sizes))

        fig = mp.figure()
        ax = fig.gca()

        ax.plot(target_cluster_sizes, "o-")
        ax.plot(source_cluster_sizes, "o-")
        ax.plot(itarget, target_cluster_sizes[itarget], "o", label="$Target$")
        ax.plot(jsource, source_cluster_sizes[jsource], "o", label="$Source$")
        ax.axhline(int(np.mean(target_cluster_sizes)), ls="--", color="k")
        ax.axhline(int(np.mean(source_cluster_sizes)), ls=":", color="k")

        ax.set_xlabel("$block$")
        ax.set_ylabel(r"$\#points$")

        fig.savefig(f"{basename}_cluster_sizes")
        mp.close(fig)

    from pytential.linalg.utils import make_index_list
    source_indices = make_index_list([source_partition.cluster_indices(jsource)])
    target_indices = make_index_list([target_partition.cluster_indices(itarget)])

    from pytential.linalg.utils import TargetAndSourceClusterList
    return TargetAndSourceClusterList(target_indices, source_indices), \
            (target_partition, itarget), \
            (source_partition, jsource)


def make_cluster_indices_single(
        actx, density_discr, *,
        source_size: int, target_size: int,
        itarget: Optional[int], jsource: Optional[int]):
    import dsplayground as ds
    nodes = ds.get_discr_nodes(density_discr)
    logger.info("%.12e", la.norm(nodes))

    # get centers
    if itarget is None:
        itarget = 0
    target_center = nodes[:, itarget]
    print(target_center)

    if jsource is None:
        jsource = ds.find_farthest_apart_node(nodes, target_center)
    source_center = nodes[:, jsource]

    # get indices
    from pytential.linalg.utils import make_index_list
    source_indices = make_index_list(
        [ds.find_nodes_around_center(nodes, source_center, source_size)])
    target_indices = make_index_list(
        [ds.find_nodes_around_center(nodes, target_center, target_size)])

    from pytential.linalg.utils import TargetAndSourceClusterList
    return TargetAndSourceClusterList(target_indices, source_indices), itarget, jsource


def make_proxies_for_collection(actx, places, mat_indices, *,
        approx_nproxy: int,
        radius_factor: float,
        dofdesc: Any,
        single_proxy_ball: bool,
        double_proxy_factor: float = 1.25,
        ):
    import dsplayground as ds

    def make_proxies(n: int, *,
            single: bool = True,
            method: str = "equidistant") -> np.ndarray:
        if single:
            return ds.make_sphere(n, method=method)
        else:
            assert double_proxy_factor >= 1
            return np.hstack([
                ds.make_sphere(n, method=method),
                double_proxy_factor * ds.make_sphere(n, method=method)
                ])

    from pytential.linalg import QBXProxyGenerator as ProxyGenerator
    proxy = ProxyGenerator(places,
            approx_nproxy=approx_nproxy,
            radius_factor=radius_factor,
            norm_type="l2",
            _generate_ref_proxies=partial(make_proxies, single=single_proxy_ball),
            )

    return proxy(actx, dofdesc, mat_indices.targets)


def make_wrangler(places, *, target, source):
    from pytential.symbolic.matrix import (
            P2PClusterMatrixBuilder, QBXClusterMatrixBuilder)

    def UnweightedP2PClusterMatrixBuilder(*args, **kwargs):      # noqa: N802
        kwargs["_weighted"] = False
        return P2PClusterMatrixBuilder(*args, **kwargs)

    def WeightedP2PClusterMatrixBuilder(*args, **kwargs):      # noqa: N802
        kwargs["_weighted"] = True
        return P2PClusterMatrixBuilder(*args, **kwargs)

    def UnweightedQBXClusterMatrixBuilder(*args, **kwargs):   # noqa: N802
        kwargs["_weighted"] = False
        return QBXClusterMatrixBuilder(*args, **kwargs)

    def WeightedQBXClusterMatrixBuilder(*args, **kwargs):     # noqa: N802
        kwargs["_weighted"] = True
        return QBXClusterMatrixBuilder(*args, **kwargs)

    from sumpy.kernel import LaplaceKernel
    kernel = LaplaceKernel(places.ambient_dim)

    from pytential import sym
    sym_sigma = sym.var("sigma")
    sym_op = sym.S(kernel, sym_sigma,
            source=source, target=target,
            qbx_forced_limit=+1)

    from pytential.linalg.skeletonization import make_skeletonization_wrangler
    wrangler = make_skeletonization_wrangler(
            places, sym_op, sym_sigma,
            domains=[source],
            context={},
            _weighted_proxy=(False, False),
            _proxy_source_cluster_builder=UnweightedP2PClusterMatrixBuilder,
            _proxy_target_cluster_builder=UnweightedQBXClusterMatrixBuilder,
            _neighbor_cluster_builder=UnweightedQBXClusterMatrixBuilder)

    return wrangler

# }}}


# {{{ run

def run_qbx_skeletonization(ctx_factory,
        itarget: Optional[int] = None,
        jsource: Optional[int] = None,
        suffix: str = "v0",
        visualize: bool = False) -> None:
    import dsplayground as ds
    actx = ds.get_cl_array_context(ctx_factory)

    np.random.seed(42)
    sli.seed(42)

    # {{{ parameters

    ambient_dim = 3

    nelements = 24
    target_order = 8
    source_ovsmp = 1
    qbx_order = 4

    nclusters = 48
    source_size = 900
    target_size = 900
    proxy_radius_factor = 1.25
    single_proxy_ball = True
    double_proxy_factor = 1.25

    basename = "qbx_skeletonization_{}d_{}_{}".format(ambient_dim, suffix, "_".join([
        str(v) for v in (
            "nelements", nelements,
            "order", target_order,
            "qbx", qbx_order,
            "factor", proxy_radius_factor)
        ]).replace(".", "_"))

    import os
    logger.info("pid:       %s", os.getpid())
    logger.info("basename:  %s", basename)

    # }}}

    # {{{ set up geometry

    places = make_geometry_collection(actx,
            nelements=nelements,
            target_order=target_order, source_ovsmp=source_ovsmp,
            qbx_order=qbx_order,
            source_name="qbx")
    density_discr = places.get_discretization("qbx")

    logger.info("nelements:     %d", density_discr.mesh.nelements)
    logger.info("ndofs:         %d", density_discr.ndofs)

    source_dd = places.auto_source.to_stage1()
    target_dd = places.auto_target.to_stage1()
    proxy_dd = source_dd.copy(geometry="proxy")

    if visualize:
        try:
            from pytential.symbolic.primitives import _mapping_max_stretch_factor
        except ImportError:
            from pytential.symbolic.primitives import \
                _simplex_mapping_max_stretch_factor as _mapping_max_stretch_factor

        from pytential import bind, sym
        normals = bind(places,
                sym.normal(places.ambient_dim).as_vector(),
                auto_where=source_dd)(actx)
        stretches = bind(places,
                _mapping_max_stretch_factor(places.ambient_dim),
                auto_where=source_dd)(actx)

        from meshmode.discretization.visualization import make_visualizer
        vis = make_visualizer(actx, density_discr, target_order, force_equidistant=True)
        vis.write_vtk_file(f"{basename}_geometry.vtu", [
            ("stretch", stretches),
            ("normal", normals)
            ], overwrite=True, use_high_order=True)

    # }}}

    # {{{ set up indices

    # mat_indices, itarget, jsource = make_cluster_indices(
    #         actx, density_discr,
    #         nclusters=nclusters,
    #         itarget=itarget, jsource=jsource)
    mat_indices, itarget, jsource = make_cluster_indices_single(
            actx, density_discr,
            source_size=source_size, target_size=target_size,
            itarget=itarget, jsource=jsource)

    if visualize:
        marker = np.zeros(density_discr.ndofs)
        marker[mat_indices.sources.indices] = 1
        marker[mat_indices.targets.indices] = -1

        from arraycontext import thaw, unflatten
        template_ary = thaw(density_discr.nodes()[0], actx)
        marker = unflatten(template_ary, actx.from_numpy(marker), actx)
        vis.write_vtk_file(f"{basename}_marker.vtu", [
            ("marker", marker),
            ], overwrite=True)

    # }}}

    # {{{ get block centers and radii

    nsources = mat_indices.sources.indices.size
    ntargets = mat_indices.targets.indices.size
    logger.info("ntargets %4d nsources %4d", ntargets, nsources)

    nodes = ds.get_discr_nodes(density_discr)
    source_nodes = mat_indices.sources.cluster_take(nodes.T, 0).T
    target_nodes = mat_indices.targets.cluster_take(nodes.T, 0).T

    source_radius, source_center = ds.get_point_radius_and_center(source_nodes)
    logger.info("sources: radius %.5e center %s", source_radius, source_center)

    target_radius, target_center = ds.get_point_radius_and_center(target_nodes)
    logger.info("targets: radius %.5e center %s", target_radius, target_center)

    max_target_radius = target_radius
    min_source_radius = np.min(
            la.norm(source_nodes - target_center.reshape(-1, 1), axis=0)
            )

    from arraycontext import flatten
    expansion_radii = bind(places,
                           sym.expansion_radii(ambient_dim),
                           auto_where=target_dd)(actx)
    expansion_radii = mat_indices.targets.cluster_take(
        actx.to_numpy(flatten(expansion_radii, actx)), 0
    )
    max_expansion_radius = np.max(expansion_radii)
    logger.info("max_expansion_radius %.5e", max_expansion_radius)

    proxy_radius = proxy_radius_factor * (max_target_radius + max_expansion_radius)
    logger.info("proxy:   radius %.5e", proxy_radius)

    # NOTE: only count the sources outside the proxy ball
    min_source_radius = max(proxy_radius, min_source_radius)
    logger.info("max_target_radius %.5e min_source_radius %.5e rho %.5e",
            max_target_radius, min_source_radius,
            max_target_radius / min_source_radius)

    # }}}

    # {{{ set up proxies

    estimate_nproxy = ds.estimate_proxies_from_id_eps(ambient_dim, 1.0e-16,
            max_target_radius, min_source_radius, proxy_radius,
            ntargets, nsources) + 512
    if single_proxy_ball:
        estimate_nproxy *= 2

    pxy = make_proxies_for_collection(actx, places, mat_indices,
            approx_nproxy=1024,
            radius_factor=proxy_radius_factor,
            dofdesc=target_dd,
            single_proxy_ball=single_proxy_ball,
            double_proxy_factor=double_proxy_factor,
            )
    proxy_radius = actx.to_numpy(pxy.radii)[0]

    logger.info("estimate_nproxy: %d nproxies %d",
                estimate_nproxy, pxy.points.shape[-1])

    wrangler = make_wrangler(places, target=target_dd, source=source_dd)

    if visualize:
        import meshmode.mesh.generation as mgen
        sphere = mgen.generate_sphere(1.0, 4,
                uniform_refinement_rounds=2,
                )

        from meshmode.mesh.processing import affine_map
        sphere = affine_map(sphere, A=proxy_radius, b=target_center)
        proxy_discr = density_discr.copy(mesh=sphere)

        from meshmode.discretization.visualization import make_visualizer
        pvis = make_visualizer(actx, proxy_discr, target_order)
        pvis.write_vtk_file(f"{basename}_proxy_geometry.vtu", [], overwrite=True)
        del pvis

    # }}}

    # {{{ errors

    # NOTE: everything in here is going to go into an `npz` file for storage
    cache = {
        # parameters
        "ambient_dim": ambient_dim,
        "nelements": nelements,
        "target_order": target_order,
        "source_ovsmp": source_ovsmp,
        "qbx_order": qbx_order,
        # skeletonization
        "nclusters": nclusters,
        "proxy_radius_factory": proxy_radius_factor,
        "ntargets": ntargets, "itarget": itarget,
        "nsources": nsources, "jsource": jsource,
        "target_center": target_center, "target_radius": target_radius,
        "source_center": source_center, "source_radius": source_radius,
        "proxy_radius": proxy_radius,
        "min_source_radius": min_source_radius,
        "max_expansion_radius": max_expansion_radius,
    }

    # {{{ error vs. id_eps

    id_eps_array = 10.0**(-np.arange(2, 16))
    # id_eps_array = 10.0**(-np.array([12]))
    rec_errors = np.empty((id_eps_array.size,))
    id_rank = np.empty(id_eps_array.size, dtype=np.int64)
    pxy_rank = np.empty(id_eps_array.size, dtype=np.int64)

    for i, id_eps in enumerate(id_eps_array):
        info = compute_target_reconstruction_error(
                actx, wrangler, places, mat_indices, pxy,
                source_dd=source_dd, target_dd=target_dd, proxy_dd=proxy_dd,
                id_eps=id_eps,
                )
        rec_errors[i] = info.error
        id_rank[i] = info.id_rank
        pxy_rank[i] = info.proxy_rank

        rho = proxy_radius / min_source_radius
        c = np.prod(info.interaction_mat.shape) / (4.0 * np.pi)

        logger.info("id_eps %.5e rec error %.5e estimate %.5e",
                    id_eps, rec_errors[i],
                    1.0e-5 * c * rho ** np.sqrt(id_rank[i]))

    if visualize:
        U, _, V = la.svd(info.error_mat)        # noqa: N806

        from arraycontext import thaw, unflatten
        template_ary = thaw(density_discr.nodes()[0], actx)

        vec = np.zeros(density_discr.ndofs)
        names_and_fields = [("normal", normals)]
        for k in range(4):
            vec[mat_indices.targets.indices] = U[:, k].ravel()
            names_and_fields.append((
                f"U_{k}", unflatten(template_ary, actx.from_numpy(vec), actx)
                ))

            vec[:] = 0.0
            vec[mat_indices.sources.indices] = V[k, :].ravel()
            names_and_fields.append((
                f"V_{k}", unflatten(template_ary, actx.from_numpy(vec), actx)
                ))

        vis.write_vtk_file(f"{basename}_svd_vectors.vtu",
                names_and_fields, overwrite=True)

        names_and_fields = [("normal", normals)]
        for k in range(4):
            density = V[k, :].ravel()

            vec[mat_indices.targets.indices] = info.rec_mat @ density
            names_and_fields.append((
                f"rec_mat_{k}", unflatten(template_ary, actx.from_numpy(vec), actx)
                ))

            vec[mat_indices.targets.indices] = info.interaction_mat @ density
            names_and_fields.append((
                f"int_mat_{k}", unflatten(template_ary, actx.from_numpy(vec), actx)
                ))

        vis.write_vtk_file(f"{basename}_svd_density.vtu",
                names_and_fields, overwrite=True)

    cache.update({
        "estimate_nproxy": pxy.points.shape[1],
        "id_eps": id_eps_array,
        "rec_errors": rec_errors,
        "id_rank": id_rank, "proxy_rank": pxy_rank,
        "interaction_mat": info.interaction_mat,
        "error_mat": info.error_mat,
        })

    # }}}

    # # {{{ error vs proxy radius

    # proxy_radius_eps = np.array([1.0e-5, 1.0e-11, 1.0e-15])
    # if single_proxy_ball:
    #     proxy_radius_factors = np.linspace(1.0, 2.0, 8)
    # else:
    #     proxy_radius_factors = np.linspace(1.0 / double_proxy_factor, 2.0, 8)
    # proxy_radius_errors = np.empty((proxy_radius_eps.size, proxy_radius_factors.size))

    # for j in range(proxy_radius_factors.size):
    #     pxy = make_proxies_for_collection(
    #             actx, places, mat_indices,
    #             approx_nproxy=estimate_nproxy,
    #             radius_factor=proxy_radius_factors[j],
    #             dofdesc=target_dd,
    #             single_proxy_ball=single_proxy_ball,
    #             source_dd=source_dd, target_dd=target_dd, proxy_dd=proxy_dd)

    #     wrangler = make_wrangler(places, target=target_dd, source=source_dd)

    #     for i in range(proxy_radius_eps.size):
    #         info = compute_target_reconstruction_error(
    #                 actx, wrangler, places, mat_indices, pxy,
    #                 source_dd=source_dd, target_dd=target_dd, proxy_dd=proxy_dd,
    #                 id_eps=proxy_radius_eps[i])

    #         proxy_radius_errors[i, j] = info.error
    #         logger.info("factor %.5e eps %.5e error %.12e",
    #                 proxy_radius_factors[j], proxy_radius_eps[i], info.error)

    # cache.update({
    #     "proxy_radius_factors": proxy_radius_factors,
    #     "proxy_radius_eps": proxy_radius_eps,
    #     "proxy_radius_errors": proxy_radius_errors,
    #     })

    # # }}}

    # # {{{ error vs model

    # nproxy_empirical = np.empty(id_eps_array.size, dtype=np.int64)
    # nproxy_model = np.empty(id_eps_array.size, dtype=np.int64)
    # id_rank = np.empty(id_eps_array.size, dtype=np.int64)

    # nproxies = 3
    # nproxy_increment = 4
    # nproxy_model_factor = 1

    # for i, id_eps in enumerate(id_eps_array):
    #     # {{{ increase nproxies until the id_eps tolerance is reached

    #     test_nproxies = []
    #     test_rec_errors = []

    #     # start from the previous value, but cap at ntargets or nsources
    #     nproxies = min(
    #             max(nproxies - 2 * nproxy_increment, 3),
    #             max(nsources, ntargets))

    #     while nproxies < 2 * max(nsources, ntargets):
    #         proxy = ProxyGenerator(places,
    #                 approx_nproxy=nproxies,
    #                 radius_factor=proxy_radius_factor,
    #                 _generate_ref_proxies=partial(
    #                 ds.make_sphere, method="fibonacci"),
    #                 )
    #         pxy = proxy(actx, target_dd, target_indices)

    #         proxy_indices = TargetAndSourceClusterList(target_indices, pxy.indices)
    #         places = places.merge({
    #             proxy_dd.geometry: pxy.as_sources(actx, qbx_order)
    #             })

    #         info = compute_target_reconstruction_error(
    #                 actx, wrangler, places, mat_indices, proxy_indices,
    #                 source_dd=source_dd, target_dd=target_dd, proxy_dd=proxy_dd,
    #                 id_eps=id_eps,
    #                 verbose=False,
    #                 )

    #         if info.error < 5 * id_eps:
    #             break

    #         test_nproxies.append(nproxies)
    #         test_rec_errors.append(info.error)

    #         logger.info("nproxy %5d id_eps %.5e got %.12e",
    #                 nproxies, id_eps, info.error)

    #         nproxies += nproxy_increment

    #     # }}}

    #     # {{{ visualize proxy error

    #     if visualize and len(test_nproxies) > 3:
    #         fig = mp.figure()
    #         ax = fig.gca()

    #         ax.semilogy(np.array(test_nproxies), np.array(test_rec_errors))
    #         ax.set_xlabel(r"$\#proxies$")
    #         ax.set_ylabel("$Error$")

    #         fig.savefig(f"{basename}_model_error_{i:02d}")
    #         mp.close(fig)

    #     # }}}

    #     nproxy_empirical[i] = nproxies
    #     nproxy_model[i] = nproxy_model_factor * ds.estimate_proxies_from_id_eps(
    #             ambient_dim, id_eps,
    #             max_target_radius, min_source_radius, proxy_radius,
    #             ntargets, nsources)
    #     id_rank[i] = info.rank

    #     logger.info("id_eps %.5e nproxy empirical %3d model %3d rank %3d / %3d",
    #             id_eps, nproxy_empirical[i], nproxy_model[i], info.rank, ntargets)

    # cache.update({
    #     "nproxy_empirical": nproxy_empirical,
    #     "nproxy_model": nproxy_model,
    #     "id_rank": id_rank,
    #     })

    # # }}}

    # {{{ save and plot

    filename = f"{basename}.npz"
    np.savez_compressed(filename, **cache)
    plot_qbx_skeletonization(filename)

    # }}}

# }}}


# {{{ plot

def plot_qbx_skeletonization(filename: str) -> None:
    import dsplayground as ds           # noqa: F401

    import pathlib
    datafile = pathlib.Path(filename)
    basename = datafile.with_suffix("")

    r = np.load(datafile)
    fig = mp.figure()

    # {{{ convergence vs id_eps

    if "rec_errors" in r:
        ax = fig.gca()
        id_eps = r["id_eps"]
        rec_errors = r["rec_errors"]

        # rho = r["proxy_radius"] / r["min_source_radius"]
        # c = 1.0e-5 * np.prod(r["interaction_mat"].shape) / (4.0 * np.pi)
        # translation_model_error = c * rho ** np.sqrt(r["id_rank"])

        ax = fig.gca()
        ax.loglog(id_eps, rec_errors, "o-")
        # ax.loglog(id_eps, translation_model_error, "o-")
        ax.loglog(id_eps, id_eps, "k--")
        ax.set_xlabel(r"$\epsilon_{id}$")
        ax.set_ylabel(r"$Relative~Error$")

        fig.savefig(f"{basename}_id_eps")
        fig.clf()

    # }}}

    # {{{ svd at lowest id_eps

    if "error_mat" in r:
        error_mat = r["error_mat"]
        u_mat, sigma, _ = la.svd(error_mat)

        ax = fig.gca()
        ax.semilogy(abs(sigma), label=fr"$\sigma_{{max}} = {np.max(sigma):.5e}$")
        ax.set_xlabel("$Target$")
        ax.set_ylabel(r"$\sigma$")
        ax.legend()

        fig.savefig(f"{basename}_svd_values")
        fig.clf()

        ax = fig.gca()
        for k in range(4):
            ax.plot(u_mat[:, k], label=f"$U_{{:, {k}}}$")
        ax.set_xlabel("$Target$")
        ax.legend()

        fig.savefig(f"{basename}_svd_uvectors")
        fig.clf()

    # }}}

    # {{{

    if "proxy_radius_factors" in r:
        proxy_radius_factors = r["proxy_radius_factors"]
        proxy_radius_eps = r["proxy_radius_eps"]
        proxy_radius_errors = r["proxy_radius_errors"]

        ax = fig.gca()

        for i in range(proxy_radius_errors.shape[0]):
            ax.semilogy(proxy_radius_factors, proxy_radius_errors[i], "o-",
                    label=fr"$\epsilon = {proxy_radius_eps[i]:.1e}$")
            ax.axhline(proxy_radius_eps[i], color="k", ls="--")

        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("$Relative~Error$")

        fig.savefig(f"{basename}_proxy_radius_factor")
        fig.clf()

    # }}}

    # {{{ rank

    if "id_rank" in r:
        nproxies = r["estimate_nproxy"]
        id_rank = r["id_rank"]
        proxy_rank = r["proxy_rank"]

        ax = fig.gca()

        ax.semilogx(id_eps, id_rank, label="$ID$")
        ax.semilogx(id_eps, proxy_rank, label="$Numerical$")
        ax.axhline(nproxies, ls="--", color="k")
        ax.set_xlabel(r"$\epsilon$")
        ax.set_ylabel("$Rank$")
        ax.legend()

        fig.savefig(f"{basename}_rank")
        fig.clf()

    # }}}

    # {{{ convergence vs model

    if "nproxy_empirical" in r:
        nproxy_empirical = r["nproxy_empirical"]
        nproxy_model = r["nproxy_model"]

        ax = fig.gca()

        ax.semilogx(id_eps, nproxy_empirical, "o-", label="$Empirical$")
        ax.semilogx(id_eps, nproxy_model, "ko-", label="$Model$")
        ax.set_xlabel(r"$\epsilon$")
        ax.set_ylabel(r"$\#~proxy$")
        ax.legend()

        fig.savefig(f"{basename}_model_vs_empirical")
        fig.clf()

    # }}}

    mp.close(fig)

# }}}


if __name__ == "__main__":
    import sys
    import pyopencl as cl
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        run_qbx_skeletonization(cl.create_some_context)
