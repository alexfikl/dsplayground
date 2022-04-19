import numpy as np

import logging
logger = logging.getLogger(__name__)


def main(ctx_factory, *,
        nelements: int = 128,
        target_order: int = 16,
        qbx_order: int = 4,
        nclusters: int = 16,
        proxy_approx_nproxy: int = 32,
        proxy_radius_factor: float = 1.01,
        norm_type="l2",
        ) -> None:
    import dsplayground as ds
    actx = ds.get_cl_array_context(ctx_factory)

    # {{{ geometry

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

    # }}}

    # {{{ proxy

    from pytential.linalg.proxy import partition_by_nodes
    max_particles_in_box = density_discr.ndofs // nclusters
    cindex = partition_by_nodes(
            actx, places, dofdesc=places.auto_source,
            max_particles_in_box=max_particles_in_box)
    logger.info("nclusters: %d (desired %d)", cindex.nclusters, nclusters)

    from pytential.linalg.proxy import ProxyGenerator, QBXProxyGenerator
    generator = ProxyGenerator(places,
            approx_nproxy=proxy_approx_nproxy,
            radius_factor=proxy_radius_factor * 1.5,
            norm_type=norm_type)
    p2p = generator(actx, places.auto_source, cindex).to_numpy(actx, True)

    generator = QBXProxyGenerator(places,
            approx_nproxy=proxy_approx_nproxy,
            radius_factor=proxy_radius_factor,
            norm_type=norm_type)
    qbx = generator(actx, places.auto_source, cindex).to_numpy(actx, True)

    # }}}

    # {{{ visualize

    from arraycontext import flatten, thaw
    nodes = actx.to_numpy(
            flatten(thaw(density_discr.nodes(), actx), actx)
            ).reshape(places.ambient_dim, -1)

    import matplotlib.pyplot as mp
    fig = mp.figure(figsize=(10, 10), dpi=300)
    ax = fig.gca()

    ax.plot(nodes[0], nodes[1])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect("equal")

    import matplotlib.patches as patch
    for i in range(cindex.nclusters):

        circle = patch.Circle(
                p2p.centers[:, i], p2p.radii[i],
                alpha=0.25)
        ax.add_patch(circle)

        circle = patch.Circle(
                qbx.centers[:, i], qbx.radii[i],
                alpha=0.25, facecolor="k")
        ax.add_patch(circle)

    filename = f"proxy_radius_{norm_type}_order_{target_order}"
    fig.savefig(filename)

    # }}}


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    import pyopencl as cl
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        main(cl.create_some_context)
