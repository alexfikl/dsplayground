from functools import partial
from typing import Optional

import numpy as np
import numpy.linalg as la

import scipy.linalg.interpolative as sli    # pylint: disable=no-name-in-module

import matplotlib.pyplot as mp

import logging
logger = logging.getLogger(__name__)


# {{{ skeletonize and compute errors

def compute_target_reconstruction_error(
        actx, wrangler, places, mat_indices, proxy_indices, *,
        source_dd, target_dd, proxy_dd, id_eps,
        verbose=True):
    # {{{ evaluate matrices

    proxy_mat = wrangler.evaluate_target_farfield(
            actx, places, 0, 0, proxy_indices,
            auto_where=(proxy_dd, target_dd))
    proxy_mat = proxy_mat.reshape(proxy_indices.block_shape(0, 0))

    interaction_mat = wrangler.evaluate_nearfield(
            actx, places, 0, 0, mat_indices,
            auto_where=(source_dd, target_dd))
    interaction_mat = interaction_mat.reshape(mat_indices.block_shape(0, 0))

    # }}}

    # {{{ skeletonize

    k, idx, proj = sli.interp_decomp(proxy_mat.T, id_eps)
    P = sli.reconstruct_interp_matrix(idx, proj).T      # noqa: N806
    idx = idx[:k]

    id_error = la.norm(proxy_mat - P @ proxy_mat[idx, :]) / la.norm(proxy_mat)
    if verbose:
        logger.info("id_rank:   %3d num_rank %3d nproxy %4d",
                idx.size, la.matrix_rank(proxy_mat, tol=id_eps), proxy_mat.shape[1])
        logger.info("id_error:  %.15e (eps %.5e)", id_error, id_eps)
        logger.info("\n")

    # }}}

    # {{{ compute reconstruction error

    rec_error = la.norm(
            interaction_mat - P @ interaction_mat[idx, :]
            ) / la.norm(interaction_mat)

    # }}}

    return rec_error, k

# }}}


# {{{ run

def run_qbx_skeletonization(ctx_factory,
        itarget: Optional[int] = None,
        jsource: Optional[int] = None,
        visualize: bool = True) -> None:
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
    target_order = 14
    source_ovsmp = 1
    qbx_order = 4

    nblocks = 24
    proxy_radius_factor = 1.5

    basename = "qbx_skeletonization_{}d_{}".format(ambient_dim, "_".join([
        str(v) for v in (
            "nelements", nelements,
            "order", target_order,
            "qbx", qbx_order,
            "factor", proxy_radius_factor)
        ]).replace(".", "_"))
    logger.info("basename: %s", basename)

    # }}}

    # {{{ set up geometry

    import meshmode.mesh.generation as mgen
    mesh = mgen.generate_torus(10, 5,
            n_major=nelements, n_minor=nelements // 2,
            order=target_order)

    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory
    from meshmode.discretization import Discretization
    pre_density_discr = Discretization(actx, mesh,
            group_factory=InterpolatoryQuadratureSimplexGroupFactory(target_order))

    from pytential.qbx import QBXLayerPotentialSource
    qbx = QBXLayerPotentialSource(pre_density_discr,
            fine_order=source_ovsmp * target_order,
            qbx_order=qbx_order,
            fmm_order=False, fmm_backend=None,
            _disable_refinement=True)

    from pytential import GeometryCollection
    places = GeometryCollection(qbx, auto_where="qbx")
    density_discr = places.get_discretization("qbx")

    logger.info("nelements:     %d", density_discr.mesh.nelements)
    logger.info("ndofs:         %d", density_discr.ndofs)

    source_dd = places.auto_source.to_stage1()
    target_dd = places.auto_target.to_stage1()
    proxy_dd = source_dd.copy(geometry="proxy")

    if visualize:
        from meshmode.discretization.visualization import make_visualizer
        vis = make_visualizer(actx, density_discr, target_order)
        vis.write_vtk_file(f"{basename}_geometry.vtu", [], overwrite=True)

    # }}}

    # {{{ set up symbolic operator

    from sumpy.kernel import LaplaceKernel
    kernel = LaplaceKernel(ambient_dim)

    from pytential import sym
    sym_sigma = sym.var("sigma")
    sym_op = sym.S(kernel, sym_sigma,
            source=source_dd, target=target_dd,
            qbx_forced_limit=+1)

    # }}}

    # {{{ set up indices

    from pytential.linalg.proxy import partition_by_nodes
    max_particles_in_box = density_discr.ndofs // nblocks
    partition = partition_by_nodes(actx, density_discr,
            max_particles_in_box=max_particles_in_box,
            tree_kind="adaptive-level-restricted")
    logger.info("nblocks %5d got %5d", nblocks, partition.nblocks)
    nblocks = partition.nblocks

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
        pass

    from pytential.linalg.utils import make_block_index_from_array
    source_indices = make_block_index_from_array(
            [partition.block_indices(jsource)])
    target_indices = make_block_index_from_array(
            [partition.block_indices(itarget)])

    from pytential.linalg.utils import MatrixBlockIndexRanges
    mat_indices = MatrixBlockIndexRanges(target_indices, source_indices)

    # }}}

    # {{{ visualize partition sizes

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

    # }}}

    # {{{ plot indices and proxies

    if visualize:
        marker = np.zeros(density_discr.ndofs)
        marker[source_indices.indices] = 1
        marker[target_indices.indices] = -1

        from arraycontext import thaw, unflatten
        template_ary = thaw(density_discr.nodes()[0], actx)
        marker = unflatten(template_ary, actx.from_numpy(marker), actx)
        vis.write_vtk_file(f"{basename}_marker.vtu", [
            ("marker", marker),
            ], overwrite=True)

    # }}}

    # {{{ get block centers and radii

    nsources = source_indices.indices.size
    ntargets = target_indices.indices.size
    logger.info("ntargets %4d nsources %4d", ntargets, nsources)

    nodes = ds.get_discr_nodes(density_discr)
    source_nodes = source_indices.block_take(nodes.T, 0).T
    target_nodes = target_indices.block_take(nodes.T, 0).T

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

    # {{{ plot proxy ball

    if visualize:
        import meshmode.mesh.generation as mgen
        sphere = mgen.generate_sphere(1.0, 4,
                uniform_refinement_rounds=2,
                )

        from meshmode.mesh.processing import affine_map
        sphere = affine_map(sphere, A=proxy_radius, b=target_center)
        proxy_discr = density_discr.copy(mesh=sphere)

        from meshmode.discretization.visualization import make_visualizer
        vis = make_visualizer(actx, proxy_discr, target_order)
        vis.write_vtk_file(f"{basename}_proxy_geometry.vtu", [], overwrite=True)

    # }}}

    # {{{ set up proxies

    estimate_nproxy = ds.estimate_proxies_from_id_eps(ambient_dim, 1.0e-16,
            max_target_radius, min_source_radius, proxy_radius,
            ntargets, nsources) * 2

    from pytential.linalg import QBXProxyGenerator as ProxyGenerator
    proxy = ProxyGenerator(places,
            approx_nproxy=estimate_nproxy,
            radius_factor=proxy_radius_factor,
            _generate_ref_proxies=partial(ds.make_sphere, method="fibonacci"),
            )
    pxy = proxy(actx, target_dd, target_indices)

    proxy_indices = MatrixBlockIndexRanges(target_indices, pxy.indices)
    places = places.merge({
        proxy_dd.geometry: pxy.as_sources(actx, qbx_order)
        })

    # }}}

    # {{{ set up wrangler

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

    from pytential.linalg.skeletonization import make_block_evaluation_wrangler
    wrangler = make_block_evaluation_wrangler(
            places, sym_op, sym_sigma,
            domains=[source_dd],
            context={},
            _weighted_farfield=(False, False),
            _source_farfield_block_builder=UnweightedFarFieldBlockBuilder,
            _target_farfield_block_builder=UnweightedNearFieldBlockBuilder,
            _nearfield_block_builder=UnweightedNearFieldBlockBuilder)

    # }}}

    # {{{ error vs. id_eps

    id_eps_array = 10.0**(-np.arange(2, 16))
    rec_errors = np.empty((id_eps_array.size,))

    for i, id_eps in enumerate(id_eps_array):
        rec_errors[i], _ = compute_target_reconstruction_error(
                actx, wrangler, places, mat_indices, proxy_indices,
                source_dd=source_dd, target_dd=target_dd, proxy_dd=proxy_dd,
                id_eps=id_eps,
                )

        logger.info("id_eps %.5e rec error %.5e",
                id_eps, rec_errors[i])

    # }}}

    # {{{ error vs model

    nproxy_empirical = np.empty(id_eps_array.size, dtype=np.int64)
    nproxy_model = np.empty(id_eps_array.size, dtype=np.int64)
    id_rank = np.empty(id_eps_array.size, dtype=np.int64)

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

    #         rec_error, rank = compute_target_reconstruction_error(
    #                 actx, wrangler, places, mat_indices, proxy_indices,
    #                 source_dd=source_dd, target_dd=target_dd, proxy_dd=proxy_dd,
    #                 id_eps=id_eps,
    #                 verbose=False,
    #                 )

    #         if rec_error < 5 * id_eps:
    #             break

    #         test_nproxies.append(nproxies)
    #         test_rec_errors.append(rec_error)

    #         logger.info("nproxy %5d id_eps %.5e got %.12e",
    #                 nproxies, id_eps, rec_error)

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
    #     id_rank[i] = rank

    #     logger.info("id_eps %.5e nproxy empirical %3d model %3d rank %3d / %3d",
    #             id_eps, nproxy_empirical[i], nproxy_model[i], rank, ntargets)

    # }}}

    # {{{ save and plot

    filename = f"{basename}.npz"
    np.savez_compressed(filename,
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
            # convergence
            id_eps=id_eps_array,
            rec_errors=rec_errors,
            # model
            nproxy_empirical=nproxy_empirical,
            nproxy_model=nproxy_model,
            id_rank=id_rank,
            )

    if visualize:
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

    # {{{ convergence vs model

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
