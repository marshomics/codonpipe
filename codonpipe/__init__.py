"""CodonPipe: end-to-end codon usage analysis pipeline."""

# ── Third-party native-logging suppression ────────────────────────────────
# Some optional/transitive dependencies in a user's environment (e.g. a
# TensorFlow-backed package pulled in indirectly) emit C++/absl/CUDA banner
# lines to stderr at import time:
#   "Could not find cuda drivers ... GPU will not be used"
#   "This TensorFlow binary is optimized to use available CPU instructions..."
#   "All log messages before absl::InitializeLog() is called are written to STDERR"
# CodonPipe itself does not import TensorFlow. These are noise, not errors.
# The C++ logging level is read from the environment *at import time* of the
# offending library, so setting it here — at the top of the CodonPipe package,
# which is imported before any transitive heavy dependency — silences the banner
# regardless of which dependency triggers it. ``setdefault`` is used so a user
# who deliberately set a verbosity level keeps it.
import os as _os

_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # 3 = errors only (hide INFO/WARNING)
_os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
_os.environ.setdefault("GLOG_minloglevel", "2")
_os.environ.setdefault("KMP_WARNINGS", "0")
# The CUDA-driver probe in TensorFlow's cudart_stub.cc writes
#   "Could not find cuda drivers ... GPU will not be used"
# (preceded by the "All log messages before absl::InitializeLog()" banner)
# straight to stderr *before* absl reads TF_CPP_MIN_LOG_LEVEL, so the level
# vars above do not suppress that specific line. The probe is triggered
# transitively (e.g. umap-learn -> numba -> a TensorFlow-backed dependency)
# during the plotting stage. Telling the process there are no visible CUDA
# devices makes the stub skip the probe entirely — correct here, since
# CodonPipe does no GPU compute. Only set when the user has not already chosen
# a GPU configuration, so an intentional GPU setup is preserved.
if "CUDA_VISIBLE_DEVICES" not in _os.environ:
    _os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
_os.environ.setdefault("TF_CPP_MIN_VLOG_LEVEL", "3")
del _os

__version__ = "0.1.0"
