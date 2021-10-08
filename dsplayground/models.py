import numpy as np


def estimate_proxies_from_id_eps(
        ambient_dim: int, id_eps: float,
        source_radius: float, target_radius: float, proxy_radius: float,
        nsources: int, ntargets: int,
        qbx_order: int = 0,
        ) -> int:
    if id_eps > 1 or id_eps < 0:
        raise ValueError("'id_eps' should be in (0, 1)")

    if source_radius > target_radius:
        raise ValueError("'source_radius' cannot be larger than 'target_radius'")

    if source_radius > proxy_radius:
        raise ValueError("'source_radius' cannot be larger than 'proxy_radius'")

    alpha = source_radius / proxy_radius
    rho = source_radius / target_radius

    if ambient_dim == 2:
        eps = id_eps \
                * (1.0 - rho) / (1.0 - alpha) \
                * 2.0 * np.pi / rho \
                * 2.0 / (2 + ntargets) / np.sqrt(ntargets)

        # NOTE: eps and rho are both < 1, so this should always be a positive number
        p = int(np.log(eps) / np.log(rho)) - qbx_order
        nproxy = min(2 * p, ntargets)
    elif ambient_dim == 3:
        eps = id_eps \
                * (1 - rho) / (1 - alpha) \
                * 12 * np.pi / (2 + ntargets) / ntargets

        p = int(np.log(eps) / np.log(rho))
        nproxy = min(p * (p + 1) // 2, ntargets)
    else:
        raise ValueError(f"unsupported dimension: {ambient_dim}")

    return nproxy


def estimate_qbx_vs_p2p_error(
        qbx_order: int, qbx_radius: float, proxy_radius: float,
        nsources: int, ntargets: int,
        ):
    if qbx_radius > proxy_radius:
        raise ValueError("'qbx_radius' should be smaller than 'proxy_radius'")

    rho = qbx_radius / (proxy_radius - qbx_radius)
    return rho**(1.5 * qbx_order) / (1 - rho)
