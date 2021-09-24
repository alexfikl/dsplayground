from typing import Any, Dict, Optional

import numpy as np

from arraycontext import ArrayContext
from sumpy.kernel import Kernel

from pytential.source import PointPotentialSource
from pytential.target import PointsTarget


def evaluate_p2p(
        actx: ArrayContext, kernel: Kernel,
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
