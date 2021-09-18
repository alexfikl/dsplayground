from dsplayground.geometry import (
    make_grid_points, make_random_points, make_axis_points, make_circle,
    as_source, as_target, affine_map,
)

__all__ = (
    "make_grid_points", "make_random_points", "make_axis_points", "make_circle",
    "as_source", "as_target", "affine_map",
)


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
    mp.rc("legend", fontsize=32)
    mp.rc("lines", linewidth=2.5, markersize=10)
    mp.rc("axes", labelsize=32, titlesize=32)
    mp.rc("xtick", labelsize=24)
    mp.rc("ytick", labelsize=24)


_initialize_matplotlib_defaults()
