from functools import partial

import numpy as np
import numpy.linalg as la

import scipy.linalg.interpolative as sli    # pylint: disable=no-name-in-module

import matplotlib.pyplot as mp
import matplotlib.patches as patch

import logging
logger = logging.getLogger(__name__)


def compute_target_reconstruction_error(
        actx, interaction_mat, kernel,
        sources, targets, proxies, *,
        id_eps: float,
        verbose: bool = True) -> float:
    proxy_mat = kernel(targets, proxies)
    k, idx, proj = sli.interp_decomp(proxy_mat.T, id_eps)

    P = sli.reconstruct_interp_matrix(idx, proj).T       # noqa: N806
    idx = idx[:k]

    id_error = la.norm(proxy_mat - P @ proxy_mat[idx, :]) / la.norm(proxy_mat)
    if verbose:
        logger.info("id_rank:   %3d num_rank %3d nproxy %4d",
                idx.size, la.matrix_rank(proxy_mat, tol=id_eps), proxies.ndofs)
        logger.info("id_error:  %.15e (eps %.5e)", id_error, id_eps)

    rec_error = la.norm(
            interaction_mat - P @ interaction_mat[idx, :]
            ) / la.norm(interaction_mat)
    if verbose:
        logger.info("rec_error: %.15e", rec_error)
        logger.info("\n")

    return rec_error, P.shape[1]


def main(ctx_factory, visualize: bool = True) -> None:
    import dsplayground as ds
    actx = ds.get_cl_array_context(ctx_factory)

    np.random.seed(42)
    sli.seed(42)

    # {{{ parameters

    ambient_dim = 2
    nsources = 512
    ntargets = 512

    proxy_radius_factor = 1.5

    max_source_radius = 1.0
    min_target_radius = 1.5

    proxy_radius = proxy_radius_factor * max_source_radius

    # }}}

    # {{{ set up geometry

    sources = ds.make_random_points_in_sphere(ambient_dim, nsources,
            rmin=0.0, rmax=max_source_radius)
    targets = ds.make_random_points_in_sphere(ambient_dim, ntargets,
            rmin=min_target_radius, rmax=min_target_radius + 0.5)

    source_radius, source_center = ds.get_point_radius_and_center(sources)
    logger.info("sources: radius %.5e center %s", source_radius, source_center)

    target_radius, target_center = ds.get_point_radius_and_center(targets)
    logger.info("targets: radius %.5e center %s", target_radius, target_center)

    def make_proxy_points(nproxies):
        if abs(proxy_radius - min_target_radius) < 0.1:
            return np.hstack([
                proxy_radius * ds.make_circle(nproxies),
                0.85 * proxy_radius * ds.make_circle(nproxies),
                ])
        else:
            return proxy_radius * ds.make_circle(nproxies)

    if visualize:
        proxies = make_proxy_points(16)

        fig = mp.figure()
        ax = fig.gca()

        scircle = patch.Circle((0.0, 0.0), max_source_radius,
                edgecolor="none", facecolor="k", alpha=0.25)
        ax.add_patch(scircle)
        tcircle = patch.Wedge((0.0, 0.0),
                min_target_radius + 0.5, 0.0, 360.0, width=0.5,
                edgecolor="none", facecolor="k", alpha=0.15)
        ax.add_patch(tcircle)
        pcircle = patch.Circle((0.0, 0.0), proxy_radius,
                edgecolor="k", facecolor="none", lw=3)
        ax.add_patch(pcircle)

        ax.plot(sources[0], sources[1], "o")
        ax.plot(targets[0], targets[1], "o")
        ax.plot(proxies[0], proxies[1], "ko")
        ax.plot(source_center[0], source_center[1], "x")
        ax.plot(target_center[0], target_center[1], "x")

        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_aspect("equal")
        ax.margins(0.05, 0.05)

        fig.savefig("p2p_multipole_error_model_geometry")
        mp.close(fig)

    sources = ds.as_source(actx, sources)
    targets = ds.as_target(actx, targets)

    # }}}

    # {{{ set up kernel evaluation

    from sumpy.kernel import LaplaceKernel
    kernel = LaplaceKernel(ambient_dim)
    kernel = partial(ds.evaluate_p2p, actx, kernel)

    interaction_mat = kernel(targets, sources)

    # }}}

    # {{{ plot error vs id_eps

    estimate_nproxies = ds.estimate_proxies_from_id_eps(ambient_dim, 1.0e-16,
            source_radius, target_radius, proxy_radius,
            nsources, ntargets) + 16

    proxies = ds.as_source(actx, make_proxy_points(estimate_nproxies))

    id_eps_array = 10.0**(-np.arange(2, 16))
    rec_errors = np.empty((id_eps_array.size,))

    for i, id_eps in enumerate(id_eps_array):
        estimate_nproxies = ds.estimate_proxies_from_id_eps(id_eps,
                source_radius, target_radius, proxy_radius,
                nsources, ntargets)

        rec_errors[i], _ = compute_target_reconstruction_error(
                actx, interaction_mat, kernel,
                sources, targets, proxies,
                id_eps=id_eps, verbose=False)
        logger.info("id_eps %.5e estimate nproxy %d rec error %.5e",
                id_eps, estimate_nproxies, rec_errors[i])

    if visualize:
        fig = mp.figure()
        ax = fig.gca()

        ax.loglog(id_eps_array, rec_errors, "o-")
        ax.loglog(id_eps_array, id_eps_array, "k--")
        ax.set_xlabel(r"$\epsilon_{id}$")
        ax.set_ylabel(r"$Relative Error$")

        fig.savefig("p2p_multipole_error_model_vs_id_eps")
        mp.close(fig)

    # }}}

    # {{{ plot proxy count model vs estimate

    nproxy_estimate = np.empty(id_eps_array.size, dtype=np.int64)
    nproxy_model = np.empty(id_eps_array.size, dtype=np.int64)

    for i, id_eps in enumerate(id_eps_array):
        nproxies = 8
        while nproxies < 2 * max(ntargets, nsources):
            proxies = ds.as_source(actx, make_proxy_points(nproxies))

            rec_error, rank = compute_target_reconstruction_error(
                    actx, interaction_mat, kernel,
                    sources, targets, proxies,
                    id_eps=id_eps, verbose=False)

            if rec_error < 5 * id_eps:
                break

            nproxies += 2

        nproxy_estimate[i] = nproxies
        nproxy_model[i] = ds.estimate_proxies_from_id_eps(ambient_dim, id_eps,
                source_radius, target_radius, proxy_radius,
                nsources, ntargets)
        logger.info("id_eps %.5e nproxy estimate %3d model %3d rank %3d / %3d",
                id_eps, nproxy_estimate[i], nproxy_model[i], rank, ntargets)

    if visualize:
        fig = mp.figure()
        ax = fig.gca()

        ax.semilogx(id_eps_array, nproxy_estimate, "o-")
        ax.semilogx(id_eps_array, nproxy_model, "ko-")
        ax.set_xlabel(r"$\epsilon$")
        ax.set_ylabel(r"$\#~proxy$")

        fig.savefig("p2p_multipole_error_model_vs_estimate")
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
