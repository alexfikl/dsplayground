from typing import Any, Callable, Dict, List, Optional

import pytest

import numpy as np
import numpy.linalg as la
import scipy.linalg.interpolative as sli

from meshmode import _acf       # noqa: F401
from arraycontext import ArrayContext
from arraycontext import pytest_generate_tests_for_array_contexts
from meshmode.array_context import PytestPyOpenCLArrayContextFactory

from pytential.source import PointPotentialSource
from pytential.target import PointsTarget

import logging
logger = logging.getLogger(__name__)

pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
    ])


# {{{ test_farfield_skeletonization

def _make_proxies(
        proxy_radius: float, proxy_center: np.ndarray,
        nproxies: int) -> List[np.ndarray]:
    import dsplayground as ds
    return [
            ds.affine_map(
                ds.make_axis_points(np.array([0, 1]), nproxies),
                b=np.array([+1.75, 0.0])),
            ds.affine_map(
                ds.make_axis_points(np.array([0, 1]), nproxies),
                b=np.array([-1.75, 0.0])),
            ds.affine_map(
                proxy_radius * ds.make_circle(nproxies),
                b=proxy_center),
            ]


def _evaluate_p2p(
        actx: ArrayContext, kernel: Any,
        targets: PointsTarget, sources: PointPotentialSource,
        context: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if context is None:
        context = {}

    from pytential import GeometryCollection
    places = GeometryCollection((sources, targets))

    from pytential import sym
    sym_sigma = sym.var("sigma")

    from pytential.symbolic.execution import _prepare_expr
    sym_op = sym.int_g_vec(kernel, sym_sigma, qbx_forced_limit=None)
    sym_op = _prepare_expr(places, sym_op)

    from pytential.symbolic.matrix import P2PMatrixBuilder
    mat = P2PMatrixBuilder(actx,
            dep_expr=sym_sigma, other_dep_exprs=[],
            dep_source=sources, dep_discr=sources,
            places=places, context=context,
            exclude_self=False, _weighted=False)

    return mat(sym_op)


@pytest.mark.parametrize("geometry", ["grid", "axis"])
def test_farfield_skeletonization(
        actx_factory: Callable[[], ArrayContext],
        geometry: str,
        visualize: bool = True) -> None:
    actx = actx_factory()

    # {{{ generate points

    ambient_dim = 2
    id_eps = 1.0e-10

    nproxies = 32
    proxy_radius = 1.25

    nsources = ntargets = 10
    source_center = np.array([0.0, 0.0])
    target_center = np.array([-4.5, 0.0])

    import dsplayground as ds
    if geometry == "grid":
        sources = ds.make_grid_points(nsources, nsources)
        targets = ds.make_grid_points(ntargets, ntargets)
    elif geometry == "axis":
        sources = ds.make_axis_points(np.array([1, 0]), nsources**2)
        targets = ds.make_axis_points(np.array([1, 0]), ntargets**2)
    else:
        raise ValueError(f"unknown geometry: {geometry}")

    sources = ds.affine_map(sources, b=source_center)
    targets = ds.affine_map(targets, b=target_center)
    proxies = _make_proxies(proxy_radius, source_center, nproxies)
    if visualize:
        import matplotlib.pyplot as mp
        fig = mp.figure()
        ax = fig.gca()

        ax.plot(sources[0], sources[1], "o")
        ax.plot(targets[0], targets[1], "o")
        for proxy in proxies:
            ax.plot(proxy[0], proxy[1], "ko")

        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_aspect("equal")
        ax.margins(0.05, 0.05)

        fig.savefig("farfield_p2p_geometry")
        mp.close(fig)

    sources = ds.as_source(actx, sources)
    targets = ds.as_target(actx, targets)

    # }}}

    # {{{ set up layer potential

    from sumpy.kernel import LaplaceKernel
    kernel = LaplaceKernel(ambient_dim)

    # }}}

    # {{{ skeletonize sources

    interaction_mat = _evaluate_p2p(actx, kernel, targets, sources)

    def _reconstruction_error(
            pxy: np.ndarray, eps: float, verbose: bool = True,
            ) -> float:
        proxy_mat = _evaluate_p2p(actx, kernel, ds.as_target(actx, pxy), sources)
        k, idx, proj = sli.interp_decomp(proxy_mat, eps)

        P = sli.reconstruct_interp_matrix(idx, proj)        # noqa: N806
        idx = idx[:k]

        id_error = la.norm(proxy_mat - proxy_mat[:, idx] @ P) / la.norm(proxy_mat)
        logger.info("id_rank:   %3d num_rank %3d nproxy %4d",
                idx.size, la.matrix_rank(proxy_mat, tol=eps), pxy.shape[-1])
        if verbose:
            logger.info("id_error:  %.15e (eps %.5e)", id_error, eps)

        rec_error = la.norm(
                interaction_mat - interaction_mat[:, idx] @ P
                ) / la.norm(interaction_mat)
        logger.info("rec_error: %.15e", rec_error)
        if verbose:
            logger.info("\n")

        return rec_error

    for proxy in proxies:
        _reconstruction_error(proxy, id_eps)

    # }}}

    # {{{ convergence vs id_eps

    id_eps_array = 10.0**(-np.arange(2, 16))
    rec_errors = np.empty((len(proxies), id_eps_array.size))

    for p, proxy in enumerate(proxies):
        for i in range(id_eps_array.size):
            rec_errors[p, i] = _reconstruction_error(
                    proxy, id_eps_array[i], verbose=False)

    if visualize:
        fig = mp.figure()
        ax = fig.gca()

        for p, name, in enumerate(["Right", "Left", "Circle"]):
            ax.loglog(id_eps_array, rec_errors[p], "o-", label=f"${name}$")

        ax.loglog(id_eps_array, id_eps_array, "k--")
        ax.set_xlabel(r"$\epsilon_{id}$")
        ax.set_ylabel(r"$Relative Error$")
        ax.legend()

        fig.savefig("farfield_p2p_reconstruction_error_vs_eps")
        mp.close(fig)

    # }}}

    # {{{ convergence vs nproxies

    nproxies_array = np.array([8, 16, 24, 32, 40, 48, 56, 64, 72, 80])
    rec_errors = np.empty((len(proxies), nproxies_array.size))

    for i in range(nproxies_array.size):
        proxies = _make_proxies(proxy_radius, source_center, nproxies_array[i])
        for p, proxy in enumerate(proxies):
            rec_errors[p, i] = _reconstruction_error(
                    proxy, id_eps, verbose=False)

    if visualize:
        fig = mp.figure()
        ax = fig.gca()

        for p, name, in enumerate(["Right", "Left", "Circle"]):
            ax.semilogy(nproxies_array, rec_errors[p], "o-", label=f"${name}$")

        ax.axhline(id_eps, color="k", ls="--")
        ax.set_xlabel(r"$\#P$")
        ax.set_ylabel(r"$Relative Error$")
        ax.legend()

        fig.savefig("farfield_p2p_reconstruction_error_vs_nproxy")
        mp.close(fig)

    # }}}

# }}}


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
