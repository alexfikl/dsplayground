from dsplayground.geometry import (
    make_grid_points, make_axis_points, make_circle, make_sphere,
    make_random_points_in_box, make_random_points_in_sphere,

    as_source, as_target, affine_map,
    get_point_radius_and_center,
)
from dsplayground.evaluation import (
    evaluate_p2p, evaluate_qbx,
    evaluate_p2p_simple,
)
from dsplayground.models import (
    estimate_proxies_from_id_eps,
    estimate_qbx_vs_p2p_error,
)

__all__ = (
    "make_grid_points", "make_axis_points", "make_circle", "make_sphere",
    "make_random_points_in_box", "make_random_points_in_sphere",

    "as_source", "as_target", "affine_map",
    "get_point_radius_and_center",

    "get_cl_array_context",

    "evaluate_p2p", "evaluate_qbx", "evaluate_p2p_simple",

    "estimate_proxies_from_id_eps",
    "estimate_qbx_vs_p2p_error",
)


# {{{ matplotlib

def _initialize_matplotlib_defaults():
    import matplotlib
    import matplotlib.pyplot as mp

    try:
        import dufte
        mp.style.use(dufte.style)
    except ImportError:
        print("'dufte' package not found")

    # FIXME: this does not seem to be a documented way to check for tex support,
    # but it works for our current usecase. it also does not catch the actual
    # issue on porter, which is the missing `cm-super` package
    usetex = matplotlib.checkdep_usetex(True)

    mp.rc("figure", figsize=(10, 10), dpi=300)
    mp.rc("figure.constrained_layout", use=True)
    mp.rc("text", usetex=usetex)
    mp.rc("legend", fontsize=24)
    mp.rc("lines", linewidth=2.5, markersize=10)
    mp.rc("axes", labelsize=32, titlesize=32)
    mp.rc("xtick", labelsize=24)
    mp.rc("ytick", labelsize=24)


_initialize_matplotlib_defaults()

# }}}


# {{{

def get_cl_array_context(factory):
    import pyopencl as cl
    import pyopencl.tools

    if factory is None:
        factory = cl.create_some_context

    if callable(factory):
        return get_cl_array_context(factory())

    from arraycontext import ArrayContext
    if isinstance(factory, ArrayContext):
        return factory

    if isinstance(factory, cl.Context):
        ctx = factory
        queue = cl.CommandQueue(ctx)
    elif isinstance(factory, cl.CommandQueue):
        queue = factory
        ctx = queue.context
    else:
        raise TypeError(type(factory).__name__)

    from meshmode.array_context import PyOpenCLArrayContext
    return PyOpenCLArrayContext(queue,
            # allocator=cl.tools.MemoryPool(cl.tools.ImmediateAllocator(queue)),
            allocator=None,
            force_device_scalars=True)

# }}}
