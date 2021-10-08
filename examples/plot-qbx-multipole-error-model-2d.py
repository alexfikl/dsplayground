from dataclasses import dataclass, replace

from functools import partial
from typing import Any, Optional

import numpy as np
import numpy.linalg as la

import scipy.linalg.interpolative as sli    # pylint: disable=no-name-in-module

import matplotlib.pyplot as mp
import matplotlib.patches as patch

from pytential import bind, sym
from pytential.linalg.utils import MatrixBlockIndexRanges
from pytools import memoize_in, memoize_on_first_arg

import logging
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InteractionInfo:
    source_dd: sym.DOFDescriptor
    target_dd: sym.DOFDescriptor
    index_set: MatrixBlockIndexRanges
    evaluate: Any

    def interaction_mat(self, places):
        @memoize_in(self, (InteractionInfo, "interaction_mat"))
        def mat():
            return self.evaluate(places,
                    index_set=self.index_set,
                    auto_where=(self.source_dd, self.target_dd))

        return mat()


@memoize_on_first_arg
def get_numpy_nodes(discr):
    from arraycontext import thaw
    from meshmode.dof_array import flatten_to_numpy
    return np.stack(
            flatten_to_numpy(
                discr._setup_actx,
                thaw(discr.nodes(), discr._setup_actx))
            )


def find_source_in_proxy_ball(
        actx, places,
        direct: InteractionInfo, proxy: InteractionInfo, *,
        proxy_center: np.ndarray, proxy_radius: float,
        ) -> InteractionInfo:
    source_dd = direct.source_dd
    index_set = direct.index_set

    assert source_dd == direct.target_dd

    discr = places.get_discretization(source_dd.geometry, source_dd.discr_stage)
    nodes = get_numpy_nodes(discr)

    source_nodes = index_set.col.block_take(nodes.T, 0).T
    mask = la.norm(
            source_nodes - proxy_center.reshape(-1, 1), axis=0
            ) < (proxy_radius - 1.0e-8)

    from pytential.linalg.utils import make_block_index_from_array
    neighbor_indices = make_block_index_from_array([
        index_set.col.indices[mask].copy()
        ])
    index_set = MatrixBlockIndexRanges(index_set.row, neighbor_indices)

    return replace(direct, index_set=index_set)


def compute_target_reconstruction_error(
        actx, places,
        direct: InteractionInfo, proxy: InteractionInfo, *,
        id_eps: float,
        proxy_center: Optional[np.ndarray] = None,
        proxy_radius: Optional[float] = None,
        verbose: bool = True) -> float:
    # {{{ evaluate proxy matrices

    if proxy_center is not None and proxy_radius is not None:
        neighbors = find_source_in_proxy_ball(
                actx, places, direct, proxy,
                proxy_center=proxy_center, proxy_radius=proxy_radius,
                )
        neighbors_mat = neighbors.interaction_mat(places)
        assert neighbors_mat.shape == neighbors.index_set.block_shape(0, 0)

        proxy_mat = proxy.interaction_mat(places)
        assert proxy_mat.shape == proxy.index_set.block_shape(0, 0)

        proxy_mat = np.hstack([neighbors_mat, proxy_mat])
    else:
        proxy_mat = proxy.interaction_mat(places)

    # }}}

    # {{{ skeletonize proxy matrix

    k, idx, proj = sli.interp_decomp(proxy_mat.T, id_eps)
    P = sli.reconstruct_interp_matrix(idx, proj).T       # noqa: N806
    idx = idx[:k]

    id_error = la.norm(proxy_mat - P @ proxy_mat[idx, :]) / la.norm(proxy_mat)
    assert id_error < 10 * id_eps, id_error

    if verbose:
        logger.info("id_rank:   %3d num_rank %3d nproxy %4d",
                idx.size, la.matrix_rank(proxy_mat, tol=id_eps), proxy_mat.shape[0])
        logger.info("id_error:  %.15e (eps %.5e)", id_error, id_eps)

    # }}}

    # {{{ compute reconstruction error

    interaction_mat = direct.interaction_mat(places)
    rec_error = la.norm(
            interaction_mat - P @ interaction_mat[idx, :]
            ) / la.norm(interaction_mat)

    if verbose:
        logger.info("rec_error: %.15e", rec_error)
        logger.info("\n")

    # }}}

    return rec_error, P.shape[1]


def main(ctx_factory,
        itarget: Optional[int] = None,
        use_p2p_proxy: bool = False,
        visualize: bool = True) -> None:
    import dsplayground as ds
    actx = ds.get_cl_array_context(ctx_factory)

    np.random.seed(42)
    sli.seed(42)

    # {{{ parameters

    ambient_dim = 2

    nelements = 64
    target_order = 16
    qbx_order = 6

    proxy_radius_factor = 1.25
    nblocks = 8

    p2p = "p2p" if use_p2p_proxy else "qbx"
    key = f"{p2p}_{nelements:04d}_{qbx_order:02d}_{proxy_radius_factor:.2f}"
    key = key.replace(".", "_")

    # }}}

    # {{{ set up geometry

    import meshmode.mesh.generation as mgen
    mesh = mgen.make_curve_mesh(
            # lambda t: 3.0 * mgen.ellipse(1.0, t),
            lambda t: 3.0 * mgen.starfish(t),
            np.linspace(0.0, 1.0, nelements + 1),
            target_order, closed=True,
            )

    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory
    from meshmode.discretization import Discretization
    pre_density_discr = Discretization(actx, mesh,
            group_factory=InterpolatoryQuadratureSimplexGroupFactory(target_order))

    from pytential.qbx import QBXLayerPotentialSource
    qbx = QBXLayerPotentialSource(pre_density_discr,
            fine_order=target_order,
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

    # }}}

    # {{{ set up indices

    from pytential.linalg.proxy import partition_by_nodes
    max_particles_in_box = density_discr.ndofs // nblocks
    partition = partition_by_nodes(actx, density_discr,
            tree_kind=None, max_particles_in_box=max_particles_in_box)
    nblocks = partition.nblocks

    from pytential.linalg.utils import make_block_index_from_array
    # NOTE: tree_kind == None just goes around the curve in order, so we
    # expect that nblocks // 2 is about opposite to 0
    if itarget is None:
        itarget = nblocks // 2

    source_indices = make_block_index_from_array(
            [partition.block_indices(0)])
    target_indices = make_block_index_from_array(
            [partition.block_indices(itarget)])

    # }}}

    # {{{ determine radii

    nodes = get_numpy_nodes(density_discr)
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
    logger.info("max_target_radius %.5e min_source_radius %.5e",
            max_target_radius, min_source_radius)

    def make_proxy_points(nproxies):
        return proxy_radius * ds.make_circle(nproxies) + target_center.reshape(2, -1)

    def make_source_proxies(nproxies):
        return ds.as_source(actx, make_proxy_points(nproxies), qbx_order=qbx_order)

    def make_proxy_indices(proxies):
        ip = make_block_index_from_array([np.arange(proxies.ndofs)])
        return MatrixBlockIndexRanges(target_indices, ip)

    # }}}

    # {{{ plot geometries

    if visualize:
        proxies = make_proxy_points(16)

        fig = mp.figure()
        ax = fig.gca()

        scircle = patch.Circle(source_center, source_radius,
                edgecolor="none", facecolor="k", alpha=0.25)
        ax.add_patch(scircle)
        tcircle = patch.Circle(target_center, target_radius,
                edgecolor="none", facecolor="k", alpha=0.25)
        ax.add_patch(tcircle)
        pcircle = patch.Circle(target_center, proxy_radius,
                edgecolor="k", facecolor="none", lw=3)
        ax.add_patch(pcircle)

        ax.plot(nodes[0], nodes[1], "-", alpha=0.25)
        ax.plot(source_nodes[0], source_nodes[1], "o", label="$Sources$")
        ax.plot(target_nodes[0], target_nodes[1], "o", label="$Targets$")
        ax.plot(proxies[0], proxies[1], "ko", label="$Proxies$")
        ax.plot(source_center[0], source_center[1], "x")
        ax.plot(target_center[0], target_center[1], "x")

        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_aspect("equal")
        ax.margins(0.05, 0.05)
        legend = ax.legend(
                bbox_to_anchor=(0, 1.02, 1.0, 0.2),
                loc="lower left", mode="expand",
                borderaxespad=0, ncol=3)

        fig.savefig(f"qbx_multipole_error_{key}_geometry",
                bbox_extra_artists=(legend,),
                bbox_inches="tight",
                )
        mp.close(fig)

    # }}}

    # {{{ set up kernel evaluation

    from sumpy.kernel import LaplaceKernel
    knl = LaplaceKernel(ambient_dim)
    if use_p2p_proxy:
        proxy_op = partial(ds.evaluate_p2p, actx, knl, context={})
    else:
        proxy_op = partial(ds.evaluate_qbx, actx, knl, context={})

    direct = InteractionInfo(
            source_dd=source_dd,
            target_dd=target_dd,
            index_set=MatrixBlockIndexRanges(target_indices, source_indices),
            evaluate=partial(ds.evaluate_qbx, actx, knl, context={})
            )

    proxy = InteractionInfo(
            source_dd=proxy_dd,
            target_dd=target_dd,
            index_set=None,
            evaluate=proxy_op)

    # }}}

    # {{{ plot error vs id_eps

    nsources = source_indices.indices.size
    ntargets = target_indices.indices.size

    estimate_nproxies = ds.estimate_proxies_from_id_eps(ambient_dim, 1.0e-16,
            max_target_radius, min_source_radius, proxy_radius,
            ntargets, nsources) + 16

    proxies = make_source_proxies(estimate_nproxies)
    places = places.merge({proxy_dd.geometry: proxies})
    proxy = replace(proxy, index_set=make_proxy_indices(proxies))

    id_eps_array = 10.0**(-np.arange(2, 16))
    rec_errors = np.empty((id_eps_array.size,))

    for i, id_eps in enumerate(id_eps_array):
        estimate_nproxies = ds.estimate_proxies_from_id_eps(ambient_dim, id_eps,
                max_target_radius, min_source_radius, proxy_radius,
                ntargets, nsources, qbx_order=qbx_order)

        rec_errors[i], _ = compute_target_reconstruction_error(
                actx, places, direct, proxy,
                proxy_center=target_center, proxy_radius=proxy_radius,
                id_eps=id_eps, verbose=False)
        logger.info("id_eps %.5e estimate nproxy %d rec error %.5e",
                id_eps, estimate_nproxies, rec_errors[i])

    from meshmode.dof_array import flatten_to_numpy
    expansion_radii = bind(places,
            sym.expansion_radii(places.ambient_dim),
            auto_where=source_dd)(actx)
    expansion_radii = flatten_to_numpy(actx, expansion_radii)
    target_expansion_radii = target_indices.block_take(expansion_radii, 0)
    qbx_radius = np.min(target_expansion_radii)

    estimate_min_id_eps = ds.estimate_qbx_vs_p2p_error(
            qbx_order, qbx_radius, proxy_radius,
            nsources=ntargets, ntargets=nsources)
    logger.info("estimate_min_id_eps: %.5e", estimate_min_id_eps)

    if visualize:
        estimate_min_id_eps = max(estimate_min_id_eps, id_eps_array[-1])

        fig = mp.figure()
        ax = fig.gca()

        ax.loglog(id_eps_array, rec_errors, "o-")
        ax.loglog(id_eps_array, id_eps_array, "k--")
        ax.axhline(estimate_min_id_eps, color="k")
        ax.set_xlabel(r"$\epsilon_{id}$")
        ax.set_ylabel(r"$Relative Error$")

        fig.savefig(f"qbx_multipole_error_{key}_model_vs_id_eps")
        mp.close(fig)

    return

    # }}}

    # {{{ plot proxy count model vs estimate

    nproxy_estimate = np.empty(id_eps_array.size, dtype=np.int64)
    nproxy_model = np.empty(id_eps_array.size, dtype=np.int64)
    id_rank = np.empty(id_eps_array.size, dtype=np.int64)

    for i, id_eps in enumerate(id_eps_array):
        # {{{ increase nproxies until the id_eps tolerance is reached

        nproxies = 3
        while nproxies < max(ntargets, nsources):
            proxies = make_source_proxies(nproxies)
            places = places.merge({proxy_dd.geometry: proxies})
            proxy = replace(proxy, index_set=make_proxy_indices(proxies))

            rec_error, rank = compute_target_reconstruction_error(
                    actx, places, direct, proxy,
                    proxy_center=target_center, proxy_radius=proxy_radius,
                    id_eps=id_eps, verbose=False)

            if rec_error < 5 * id_eps:
                break

            nproxies += 2

        # }}}

        nproxy_estimate[i] = nproxies
        nproxy_model[i] = ds.estimate_proxies_from_id_eps(ambient_dim, id_eps,
                max_target_radius, min_source_radius, proxy_radius,
                ntargets, nsources,
                qbx_order=qbx_order)
        id_rank[i] = rank

        logger.info("id_eps %.5e nproxy estimate %3d model %3d rank %3d / %3d",
                id_eps, nproxy_estimate[i], nproxy_model[i], rank, ntargets)

    if visualize:
        fig = mp.figure()
        ax = fig.gca()

        ax.semilogx(id_eps_array, nproxy_estimate, "o-", label="$Estimate$")
        # ax.semilogx(id_eps_array, id_rank, "o-", label="$Rank$")
        ax.semilogx(id_eps_array, nproxy_model, "ko-", label="$Model$")
        ax.set_xlabel(r"$\epsilon$")
        ax.set_ylabel(r"$\#~proxy$")
        ax.legend()

        fig.savefig(f"qbx_multipole_error_{key}_model_vs_estimate")
        mp.close(fig)

    # }}}


if __name__ == "__main__":
    import sys
    import pyopencl as cl
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        main(cl._csc)
