"""Back-compat shim. All logic lives in :mod:`hg_ds_evals.preprocessing.traces`."""
from hg_ds_evals.preprocessing.traces import *  # noqa: F401,F403
from hg_ds_evals.preprocessing.traces import (  # noqa: F401
    _CandidateDocument,
    _extract_user_query,
    parse_trace_skkb as parse_trace,
)
