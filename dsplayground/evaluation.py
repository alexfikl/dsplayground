from typing import Any, Dict, Optional

import numpy as np

from arraycontext import ArrayContext
from sumpy.kernel import Kernel

from pytential import GeometryCollection
from pytential.source import PointPotentialSource
from pytential.target import PointsTarget
from pytential.linalg.utils import TargetAndSourceClusterList


# {{{ p2p

def evaluate_p2p(
        actx: ArrayContext, kernel: Kernel, places: GeometryCollection,
        auto_where: Optional[Any] = None,
        tgt_src_index: Optional[TargetAndSourceClusterList] = None,
        context: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if context is None:
        context = {}

    # {{{ symbolic

    from pytential import sym
    sym_sigma = sym.var("sigma")

    from pytential.symbolic.execution import _prepare_auto_where
    auto_where = _prepare_auto_where(auto_where, places)
    sym_op = sym.int_g_vec(kernel, sym_sigma,
            source=auto_where[0], target=auto_where[1],
            qbx_forced_limit=None)

    # }}}

    # {{{ evaluate

    source = auto_where[0]
    source_discr = places.get_discretization(source.geometry, source.discr_stage)

    if tgt_src_index is not None:
        assert tgt_src_index.nclusters == 1
        from pytential.symbolic.matrix import P2PClusterMatrixBuilder
        mat = P2PClusterMatrixBuilder(actx,
                dep_expr=sym_sigma, other_dep_exprs=[],
                dep_source=source_discr, dep_discr=source_discr,
                places=places, tgt_src_index=tgt_src_index, context=context,
                exclude_self=False, _weighted=False,
                )(sym_op)

        mat = mat.reshape(tgt_src_index.cluster_shape(0, 0))
    else:
        from pytential.symbolic.matrix import P2PMatrixBuilder
        mat = P2PMatrixBuilder(actx,
                dep_expr=sym_sigma, other_dep_exprs=[],
                dep_source=source_discr, dep_discr=source_discr,
                places=places, context=context,
                exclude_self=False, _weighted=False,
                )(sym_op)

    # }}}

    return mat


def evaluate_p2p_simple(
        actx: ArrayContext, kernel: Kernel,
        targets: PointsTarget, sources: PointPotentialSource,
        context: Optional[Dict[str, Any]] = None) -> np.ndarray:
    places = GeometryCollection((sources, targets))

    return evaluate_p2p(actx, kernel, places,
            auto_where=places.auto_where,
            context=context)

# }}}


# {{{ qbx

def evaluate_qbx(
        actx: ArrayContext, kernel: Kernel, places: GeometryCollection,
        auto_where: Optional[Any] = None,
        tgt_src_index: Optional[TargetAndSourceClusterList] = None,
        context: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if context is None:
        context = {}

    # {{{ symbolic

    from pytential import sym
    sym_sigma = sym.var("sigma")

    from pytential.symbolic.execution import _prepare_auto_where
    source, target = _prepare_auto_where(auto_where, places)
    sym_op = sym.int_g_vec(kernel, sym_sigma,
            source=source, target=target,
            qbx_forced_limit=-1)

    # }}}

    # {{{ evaluate

    dep_source = places.get_geometry(source.geometry)
    dep_discr = places.get_discretization(source.geometry, source.discr_stage)

    if tgt_src_index is not None:
        assert tgt_src_index.nclusters == 1
        from pytential.symbolic.matrix import QBXClusterMatrixBuilder
        mat = QBXClusterMatrixBuilder(actx,
                dep_expr=sym_sigma, other_dep_exprs=[],
                dep_source=dep_source, dep_discr=dep_discr,
                places=places, tgt_src_index=tgt_src_index, context=context,
                _weighted=False
                )(sym_op)

        mat = mat.reshape(tgt_src_index.cluster_shape(0, 0))
    else:
        from pytential.symbolic.matrix import MatrixBuilder
        mat = MatrixBuilder(actx,
                dep_expr=sym_sigma, other_dep_exprs=[],
                dep_source=dep_source, dep_discr=dep_discr,
                places=places, context=context,
                _weighted=False
                )(sym_op)

    # }}}

    return mat

# }}}
