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

    return rec_error, P.shape[1]

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

    ambient_dim = 3

    nelements = 24
    target_order = 16
    source_ovsmp = 1
    qbx_order = 4

    nblocks = 12
    proxy_radius_factor = 1.5

    basename = "qbx_skeletonization_{}d_{}".format(ambient_dim, "_".join([
        str(v) for v in (
            "order", target_order,
            "qbx", qbx_order,
            "factor", proxy_radius_factor)
        ]).replace(".", "_"))

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
        vis = make_visualizer(actx, density_discr)
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
            max_particles_in_box=max_particles_in_box)
    nblocks = partition.nblocks

    if itarget is None:
        itarget = 0

    if jsource is None:
        jsource = ds.find_farthest_apart_block(
                actx, density_discr, partition, itarget)

    from pytential.linalg.utils import make_block_index_from_array
    source_indices = make_block_index_from_array(
            [partition.block_indices(jsource)])
    target_indices = make_block_index_from_array(
            [partition.block_indices(itarget)])

    from pytential.linalg.utils import MatrixBlockIndexRanges
    mat_indices = MatrixBlockIndexRanges(target_indices, source_indices)

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
    logger.info("min_target_radius %.5e min_source_radius %.5e",
            max_target_radius, min_source_radius)

    # }}}

    # {{{ set up proxies

    estimate_nproxy = ds.estimate_proxies_from_id_eps(ambient_dim, 1.0e-16,
            max_target_radius, min_source_radius, proxy_radius,
            ntargets, nsources) * 2

    from pytential.linalg import QBXProxyGenerator as ProxyGenerator
    proxy = ProxyGenerator(places,
            approx_nproxy=estimate_nproxy,
            radius_factor=proxy_radius_factor)
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
            )

    if visualize:
        plot_qbx_skeletonization(filename)

    # }}}

# }}}


# {{{ plot

def plot_qbx_skeletonization(filename: str) -> None:
    import pathlib
    datafile = pathlib.Path(filename)
    basename = datafile.with_suffix("")

    r = np.load(datafile)
    fig = mp.figure()

    # {{{ convergence

    ax = fig.gca()

    id_eps = r["id_eps"]
    rec_errors = r["rec_errors"]

    ax = fig.gca()

    ax.loglog(id_eps, rec_errors, "o-")
    ax.loglog(id_eps, id_eps, "k--")
    ax.set_xlabel(r"$\epsilon_{id}$")
    ax.set_ylabel(r"$Relative~Error$")

    fig.savefig(f"{basename}_id_eps")

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
