"""
- Run target skeletonization on the interaction between two point clouds.
- Compare empirical and analytic estimates for the proxy count.
"""

from functools import partial

import numpy as np
import numpy.linalg as la

import scipy.linalg.interpolative as sli    # pylint: disable=no-name-in-module

import matplotlib.pyplot as mp

import logging
logger = logging.getLogger(__name__)


# {{{ run

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


def run_error_model(ctx_factory, visualize: bool = True) -> None:
    import dsplayground as ds
    actx = ds.get_cl_array_context(ctx_factory)

    np.random.seed(42)
    sli.seed(42)

    # {{{ parameters

    ambient_dim = 3
    nsources = 512
    ntargets = 512

    proxy_radius_factor = 1.5

    max_target_radius = 1.0
    min_source_radius = 2.5

    proxy_radius = proxy_radius_factor * max_target_radius

    # }}}

    # {{{ set up geometry

    targets = ds.make_random_points_in_sphere(ambient_dim, ntargets,
            rmin=0.0, rmax=max_target_radius)
    sources = ds.make_random_points_in_sphere(ambient_dim, nsources,
            rmin=min_source_radius, rmax=min_source_radius + 0.5)

    source_radius, source_center = ds.get_point_radius_and_center(sources)
    logger.info("sources: radius %.5e center %s", source_radius, source_center)

    target_radius, target_center = ds.get_point_radius_and_center(targets)
    logger.info("targets: radius %.5e center %s", target_radius, target_center)

    def make_proxy_points(nproxies):
        if abs(proxy_radius - min_source_radius) < 0.1:
            # NOTE: if the sources are really close to the proxies / targets,
            # the skeletonization does a pretty bad job. to counter that, we
            # insert another ring of proxy points inside the first one
            return np.hstack([
                proxy_radius * ds.make_sphere(nproxies),
                0.85 * proxy_radius * ds.make_sphere(nproxies),
                ])
        else:
            return proxy_radius * ds.make_sphere(nproxies)

    sources = ds.as_source(actx, sources)
    targets = ds.as_target(actx, targets)

    # }}}

    # {{{ set up kernel evaluation

    from sumpy.kernel import LaplaceKernel
    kernel = LaplaceKernel(ambient_dim)
    kernel = partial(ds.evaluate_p2p_simple, actx, kernel)

    interaction_mat = kernel(targets, sources)

    # }}}

    # {{{ plot error vs id_eps

    nproxy_model = ds.estimate_proxies_from_id_eps(ambient_dim, 1.0e-16,
            target_radius, source_radius, proxy_radius,
            ntargets, nsources) + 16

    proxies = ds.as_source(actx, make_proxy_points(nproxy_model))

    id_eps_array = 10.0**(-np.arange(2, 16))
    rec_errors = np.empty((id_eps_array.size,))

    for i, id_eps in enumerate(id_eps_array):
        nproxy_model = ds.estimate_proxies_from_id_eps(ambient_dim, id_eps,
                target_radius, source_radius, proxy_radius,
                ntargets, nsources)

        rec_errors[i], _ = compute_target_reconstruction_error(
                actx, interaction_mat, kernel,
                sources, targets, proxies,
                id_eps=id_eps, verbose=False)
        logger.info("id_eps %.5e model nproxy %d rec error %.5e",
                id_eps, nproxy_model, rec_errors[i])

    # }}}

    # {{{ plot proxy count model vs estimate

    nproxy_empirical = np.empty(id_eps_array.size, dtype=np.int64)
    nproxy_model = np.empty(id_eps_array.size, dtype=np.int64)
    id_rank = np.empty(id_eps_array.size, dtype=np.int64)

    nproxies = 3
    for i, id_eps in enumerate(id_eps_array):
        # {{{ increase nproxies until the id_eps tolerance is reached

        nproxies = max(nproxies - 2, 3)
        while nproxies < 2 * max(ntargets, nsources):
            proxies = ds.as_source(actx, make_proxy_points(nproxies))

            rec_error, rank = compute_target_reconstruction_error(
                    actx, interaction_mat, kernel,
                    sources, targets, proxies,
                    id_eps=id_eps, verbose=False)

            if rec_error < 5 * id_eps:
                break

            nproxies += 2

        # }}}

        nproxy_empirical[i] = nproxies
        nproxy_model[i] = ds.estimate_proxies_from_id_eps(ambient_dim, id_eps,
                target_radius, source_radius, proxy_radius,
                ntargets, nsources)
        id_rank[i] = rank

        logger.info("id_eps %.5e nproxy empirical %3d model %3d rank %3d / %3d",
                id_eps, nproxy_empirical[i], nproxy_model[i], rank, ntargets)

    # }}}

    # {{{ write and visualize

    filename = "p2p_model_{}d_{}.npz".format(ambient_dim, "_".join([
        str(v) for v in (
            "ntargets", ntargets,
            "nsources", nsources,
            "factor", proxy_radius_factor)
        ]).replace(".", "_"))

    np.savez_compressed(filename,
            # parameters
            ambient_dim=ambient_dim,
            nsources=nsources,
            ntargets=ntargets,
            proxy_radius_factor=proxy_radius_factor,
            max_target_radius=max_target_radius,
            min_source_radius=min_source_radius,
            # geometry
            sources=actx.to_numpy(sources.nodes()),
            targets=actx.to_numpy(targets.nodes()),
            proxies=make_proxy_points(32),
            # convergence
            id_eps=id_eps_array,
            rec_errors=rec_errors,
            rec_nproxy=nproxy_model,
            # model
            nproxy_empirical=nproxy_empirical,
            nproxy_model=nproxy_model,
            id_rank=id_rank,
            )

    if visualize:
        plot_error_model(filename)

    # }}}

# }}}


# {{{

def plot_error_model(datafile: str) -> None:
    import pathlib
    datafile = pathlib.Path(datafile)
    basename = datafile.with_suffix("")

    r = np.load(datafile)
    fig = mp.figure()

    # {{{ geometry

    sources = r["sources"]
    targets = r["targets"]
    proxies = r["proxies"]

    ax = fig.add_subplot(111, projection="3d")

    ax.plot(sources[0], sources[1], sources[2], "o", label="$Sources$")
    ax.plot(targets[0], targets[1], targets[2], "o", label="$Targets$")
    ax.plot(proxies[0], proxies[1], proxies[2], "ko", label="$Proxies$")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.margins(0.05, 0.05, 0.05)
    legend = ax.legend(
            bbox_to_anchor=(0, 1.02, 1.0, 0.2),
            loc="lower left", mode="expand",
            borderaxespad=0, ncol=3)

    fig.savefig(f"{basename}_geometry",
            bbox_extra_artists=(legend,),
            bbox_inches="tight",
            )
    fig.clf()

    # }}}

    # {{{ convergence errors

    id_eps = r["id_eps"]
    rec_errors = r["rec_errors"]

    ax = fig.gca()

    ax.loglog(id_eps, rec_errors, "o-")
    ax.loglog(id_eps, id_eps, "k--")
    ax.set_xlabel(r"$\epsilon_{id}$")
    ax.set_ylabel(r"$Relative Error$")

    fig.savefig(f"{basename}_id_eps")
    fig.clf()

    # }}}

    # {{{ model vs empirical

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

# }}}


if __name__ == "__main__":
    import sys
    import pyopencl as cl
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        run_error_model(cl.create_some_context)
