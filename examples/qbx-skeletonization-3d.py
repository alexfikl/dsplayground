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
    rank: int
    interaction_mat: np.ndarray
    rec_mat: np.ndarray
    error_mat: np.ndarray


def rnorm(x, y, ord=None):
    ynorm = la.norm(y, ord=ord)
    if ynorm < 1.0e-15:
        ynorm = 1

    return la.norm(x - y, ord=ord) / ynorm


def compute_target_reconstruction_error(
        actx, wrangler, places, mat_indices, pxy, *,
        source_dd, target_dd, proxy_dd, id_eps,
        ord=2, verbose=True):
    # {{{ evaluate matrices

    from pytools import memoize_in

    @memoize_in(mat_indices, (compute_target_reconstruction_error, "interaction"))
    def _interaction_mat():
        mat = wrangler._evaluate_nearfield(
            actx, places, mat_indices,
            ibrow=0, ibcol=0)
        return mat.reshape(mat_indices.block_shape(0, 0))

    proxy_mat, pxyindices = wrangler.evaluate_target_farfield(
            actx, places, pxy, None,
            ibrow=0, ibcol=0)
    proxy_mat = proxy_mat.reshape(pxyindices.block_shape(0, 0))
    interaction_mat = _interaction_mat()

    # }}}

    # {{{ skeletonize

    k, idx, proj = sli.interp_decomp(proxy_mat.T, id_eps)
    P = sli.reconstruct_interp_matrix(idx, proj).T      # noqa: N806
    idx = idx[:k]

    id_error = rnorm(proxy_mat, P @ proxy_mat[idx, :], ord=ord)
    if verbose:
        if proxy_mat.shape[0] < 10000000:
            rank = la.matrix_rank(proxy_mat, tol=id_eps)
        else:
            rank = -1

        logger.info("id_rank:   %3d num_rank %3d nproxy %4d",
                idx.size, rank, proxy_mat.shape[1])
        logger.info("id_error:  %.15e (eps %.5e)", id_error, id_eps)
        logger.info("\n")

    # }}}

    # {{{ compute reconstruction error

    rec_mat = P @ interaction_mat[idx, :]
    error_mat = interaction_mat - rec_mat
    rec_error = rnorm(rec_mat, interaction_mat, ord=ord)

    # }}}

    return ErrorInfo(
            error=rec_error, rank=k,
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
            Mesh.CharacteristicLengthMax = 0.05;
            Mesh.HighOrderOptimize = 1;
            Mesh.Algorithm = 1;

            SetFactory("OpenCASCADE");
            Sphere(1) = {0, 0, 0, 0.5};
            """,
            "geo")
    elif issubclass(cls, TensorProductElementGroup):
        script = ScriptSource(
            """
            Mesh.CharacteristicLengthMax = 0.05;
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


def make_block_indices(
        actx, density_discr, *,
        nblocks: int,
        itarget: Optional[int], jsource: Optional[int],
        by_nodes: bool = True,
        ):
    if by_nodes:
        from pytential.linalg.proxy import partition_by_nodes
        max_particles_in_box = density_discr.ndofs // nblocks
        partition = partition_by_nodes(actx, density_discr,
                max_particles_in_box=max_particles_in_box,
                tree_kind="adaptive-level-restricted")
    else:
        from pytential.linalg.proxy import partition_by_elements
        max_particles_in_box = density_discr.mesh.nelements // nblocks
        partition = partition_by_elements(actx, density_discr,
                max_particles_in_box=max_particles_in_box,
                tree_kind="adaptive-level-restricted")

    logger.info("nblocks %5d got %5d", nblocks, partition.nblocks)

    import dsplayground as ds
    if itarget is None and jsource is None:
        itarget = 0
        jsource = ds.find_farthest_apart_block(
                actx, density_discr, partition, itarget)
    elif itarget is None and jsource is not None:
        itarget = ds.find_farthest_apart_block(
                actx, density_discr, partition, jsource)
    elif itarget is not None and jsource is None:
        jsource = ds.find_farthest_apart_block(
                actx, density_discr, partition, itarget)
    else:
        assert itarget is not None and 0 <= itarget < partition.nblocks
        assert jsource is not None and 0 <= jsource < partition.nblocks

    from pytential.linalg.utils import make_block_index_from_array
    source_indices = make_block_index_from_array(
            [partition.block_indices(jsource)])
    target_indices = make_block_index_from_array(
            [partition.block_indices(itarget)])

    from pytential.linalg.utils import MatrixBlockIndexRanges
    return MatrixBlockIndexRanges(target_indices, source_indices), \
            partition, itarget, jsource


def make_proxies_for_collection(actx, places, mat_indices, *,
        approx_nproxy: int,
        radius_factor: float,
        dofdesc: Any,
        single_proxy_ball: bool,
        double_proxy_factor: float = 0.8,
        ):
    import dsplayground as ds

    def make_proxies(n: int, *,
            single: bool = True,
            method: str = "equidistant") -> np.ndarray:
        if single:
            return ds.make_sphere(n, method=method)
        else:
            assert radius_factor >= 1
            return np.hstack([
                ds.make_sphere(n, method=method),
                double_proxy_factor * ds.make_sphere(n, method=method)
                ])

    from pytential.linalg import QBXProxyGenerator as ProxyGenerator
    proxy = ProxyGenerator(places,
            approx_nproxy=approx_nproxy,
            radius_factor=radius_factor,
            _generate_ref_proxies=partial(make_proxies, single=single_proxy_ball),
            )

    return proxy(actx, dofdesc, mat_indices.row)


def make_wrangler(places, *, target, source):
    from pytential.symbolic.matrix import FarFieldBlockBuilder, NearFieldBlockBuilder

    def UnweightedFarFieldBlockBuilder(*args, **kwargs):      # noqa: N802
        kwargs["_weighted"] = False
        return FarFieldBlockBuilder(*args, **kwargs)

    def WeightedFarFieldBlockBuilder(*args, **kwargs):      # noqa: N802
        kwargs["_weighted"] = True
        return FarFieldBlockBuilder(*args, **kwargs)

    def UnweightedNearFieldBlockBuilder(*args, **kwargs):   # noqa: N802
        kwargs["_weighted"] = False
        return NearFieldBlockBuilder(*args, **kwargs)

    def WeightedNearFieldBlockBuilder(*args, **kwargs):     # noqa: N802
        kwargs["_weighted"] = True
        return NearFieldBlockBuilder(*args, **kwargs)

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
            _weighted_farfield=(False, False),
            _source_farfield_block_builder=UnweightedFarFieldBlockBuilder,
            _target_farfield_block_builder=UnweightedNearFieldBlockBuilder,
            _nearfield_block_builder=UnweightedNearFieldBlockBuilder)

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

    # This case seems to bottom out::
    #   nelements = 24, target_order = 16, qbx_order = 4,
    #   proxy_radius_factor = 1.5

    ambient_dim = 3

    nelements = 24
    target_order = 8
    source_ovsmp = 1
    qbx_order = 4

    nblocks = 44
    proxy_radius_factor = 1.25
    single_proxy_ball = True
    double_proxy_factor = 0.8

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

    mat_indices, partition, itarget, jsource = make_block_indices(
            actx, density_discr,
            nblocks=nblocks,
            itarget=itarget, jsource=jsource)
    nblocks = partition.nblocks

    if visualize:
        block_sizes = np.array([partition.block_size(i) for i in range(nblocks)])
        logger.info("block sizes: %s", list(block_sizes))

        fig = mp.figure()
        ax = fig.gca()

        ax.plot(block_sizes, "o-")
        ax.plot(itarget, block_sizes[itarget], "o", label="$Target$")
        ax.plot(jsource, block_sizes[jsource], "o", label="$Source$")
        ax.axhline(int(np.mean(block_sizes)), ls="--", color="k")

        ax.set_xlabel("$block$")
        ax.set_ylabel(r"$\#points$")

        fig.savefig(f"{basename}_block_sizes")
        mp.close(fig)

    if visualize:
        marker = np.zeros(density_discr.ndofs)
        marker[mat_indices.col.indices] = 1
        marker[mat_indices.row.indices] = -1

        from arraycontext import thaw, unflatten
        template_ary = thaw(density_discr.nodes()[0], actx)
        marker = unflatten(template_ary, actx.from_numpy(marker), actx)
        vis.write_vtk_file(f"{basename}_marker.vtu", [
            ("marker", marker),
            ], overwrite=True)

    # }}}

    # {{{ get block centers and radii

    nsources = mat_indices.col.indices.size
    ntargets = mat_indices.row.indices.size
    logger.info("ntargets %4d nsources %4d", ntargets, nsources)

    nodes = ds.get_discr_nodes(density_discr)
    source_nodes = mat_indices.col.block_take(nodes.T, 0).T
    target_nodes = mat_indices.row.block_take(nodes.T, 0).T

    source_radius, source_center = ds.get_point_radius_and_center(source_nodes)
    logger.info("sources: radius %.5e center %s", source_radius, source_center)

    target_radius, target_center = ds.get_point_radius_and_center(target_nodes)
    logger.info("targets: radius %.5e center %s", target_radius, target_center)

    proxy_radius = proxy_radius_factor * target_radius
    logger.info("proxy:   radius %.5e", proxy_radius)

    max_target_radius = target_radius
    min_source_radius = np.min(
            la.norm(source_nodes - target_center.reshape(-1, 1), axis=0)
            )
    # NOTE: only count the sources outside the proxy ball
    min_source_radius = max(proxy_radius, min_source_radius)
    logger.info("max_target_radius %.5e min_source_radius %.5e rho %.5e",
            max_target_radius, min_source_radius,
            max_target_radius / min_source_radius)

    # }}}

    # {{{ set up proxies

    estimate_nproxy = ds.estimate_proxies_from_id_eps(ambient_dim, 1.0e-16,
            max_target_radius, min_source_radius, proxy_radius,
            ntargets, nsources)
    if single_proxy_ball:
        estimate_nproxy *= 4
    logger.info("estimate_nproxy: %d", estimate_nproxy)

    pxy = make_proxies_for_collection(actx, places, mat_indices,
            approx_nproxy=estimate_nproxy,
            radius_factor=proxy_radius_factor,
            dofdesc=target_dd,
            single_proxy_ball=single_proxy_ball,
            double_proxy_factor=double_proxy_factor,
            )

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
    cache = dict(
            # parameters
            ambient_dim=ambient_dim,
            nelements=nelements,
            target_order=target_order,
            source_ovsmp=source_ovsmp,
            qbx_order=qbx_order,
            # skeletonization
            nblocks=nblocks,
            proxy_radius_factory=proxy_radius_factor,
            ntargets=ntargets, itarget=itarget,
            nsources=nsources, jsource=jsource,
            target_center=target_center, target_radius=target_radius,
            source_center=source_center, source_radius=source_radius,
            proxy_radius=proxy_radius,
            )

    # {{{ error vs. id_eps

    id_eps_array = 10.0**(-np.arange(2, 16))
    # id_eps_array = 10.0**(-np.array([12]))
    rec_errors = np.empty((id_eps_array.size,))

    for i, id_eps in enumerate(id_eps_array):
        info = compute_target_reconstruction_error(
                actx, wrangler, places, mat_indices, pxy,
                source_dd=source_dd, target_dd=target_dd, proxy_dd=proxy_dd,
                id_eps=id_eps,
                )
        rec_errors[i] = info.error

        logger.info("id_eps %.5e rec error %.5e", id_eps, rec_errors[i])

    if visualize:
        U, sigma, V = la.svd(info.error_mat)        # noqa: N806

        from arraycontext import thaw, unflatten
        template_ary = thaw(density_discr.nodes()[0], actx)

        vec = np.zeros(density_discr.ndofs)
        names_and_fields = [("normal", normals)]
        for k in range(4):
            vec[mat_indices.row.indices] = U[:, k].ravel()
            names_and_fields.append((
                f"U_{k}", unflatten(template_ary, actx.from_numpy(vec), actx)
                ))

            vec[:] = 0.0
            vec[mat_indices.col.indices] = V[k, :].ravel()
            names_and_fields.append((
                f"V_{k}", unflatten(template_ary, actx.from_numpy(vec), actx)
                ))

        vis.write_vtk_file(f"{basename}_svd_vectors.vtu",
                names_and_fields, overwrite=True)

        names_and_fields = [("normal", normals)]
        for k in range(4):
            density = V[k, :].ravel()

            vec[mat_indices.row.indices] = info.rec_mat @ density
            names_and_fields.append((
                f"rec_mat_{k}", unflatten(template_ary, actx.from_numpy(vec), actx)
                ))

            vec[mat_indices.row.indices] = info.interaction_mat @ density
            names_and_fields.append((
                f"int_mat_{k}", unflatten(template_ary, actx.from_numpy(vec), actx)
                ))

        vis.write_vtk_file(f"{basename}_svd_density.vtu",
                names_and_fields, overwrite=True)

    cache.update({
        "id_eps": id_eps_array,
        "rec_errors": rec_errors,
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

    #         proxy_indices = MatrixBlockIndexRanges(target_indices, pxy.indices)
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

        ax = fig.gca()

        ax.loglog(id_eps, rec_errors, "o-")
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

    # {{{ convergence vs model

    if "nproxy_empirical" in r:
        nproxy_empirical = r["nproxy_empirical"]
        nproxy_model = r["nproxy_model"]

        ax = fig.gca()

        ax.semilogx(id_eps, nproxy_empirical, "o-", label="$Empirical$")
        # ax.semilogx(id_eps_array, id_rank, "o-", label="$Rank$")
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
        run_qbx_skeletonization(cl._csc)
