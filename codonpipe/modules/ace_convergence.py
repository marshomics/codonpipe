"""DEPRECATED: ACE convergence has been replaced by Mahalanobis distance-based clustering.

See codonpipe.modules.mahal_clustering for the replacement implementation.
This file is retained only for backward compatibility and will be removed
in a future release.
"""

import warnings

warnings.warn(
    "ace_convergence module is deprecated. Use mahal_clustering instead.",
    DeprecationWarning,
    stacklevel=2,
)


def run_ace_convergence(**kwargs):
    """Deprecated. Raises RuntimeError directing users to mahal_clustering."""
    raise RuntimeError(
        "ACE convergence has been replaced by Mahalanobis distance-based clustering. "
        "Use codonpipe.modules.mahal_clustering.run_mahal_clustering() instead."
    )
