# hg_ds_evals/rubrics/dimensions/catalog.py
"""
Reusable dimension catalog for evaluation rubrics.

This module provides pre-defined Dimension objects that can be composed
into Rubric instances. Import dimensions from here to build custom rubrics.

Usage:
    from hg_ds_evals.rubrics.dimensions.catalog import (
        USER_QUERY_CLARITY,
        TOPIC_RELEVANCE,
        INFORMATION_COMPLETENESS,
    )
    
    # Use directly
    rubric = Rubric(
        metadata=...,
        dimensions=(USER_QUERY_CLARITY, TOPIC_RELEVANCE),
    )
    
    # Or customize weight
    rubric = Rubric(
        metadata=...,
        dimensions=(TOPIC_RELEVANCE.with_weight(2.0),),
    )

Naming Convention:
    - Dimension constants are UPPER_SNAKE_CASE
    - IDs are lower_snake_case matching the constant name
"""

from __future__ import annotations

from pathlib import Path

from hg_ds_evals.core.types import Dimension, ScoreLevel


# =============================================================================
# STANDARD SCALE
# =============================================================================

STANDARD_SCALE_3PT = (
    ScoreLevel(score=0, label="bad", description="Does not meet criteria"),
    ScoreLevel(score=1, label="partial", description="Partially meets criteria"),
    ScoreLevel(score=2, label="good", description="Fully meets criteria"),
)

# =============================================================================
# QUERY UNDERSTANDING DIMENSIONS
# =============================================================================

USER_QUERY_CLARITY = Dimension(
    id="user_query_clarity",
    name="User Query Clarity",
    description="Is the user_query (prefer user_query_contexted) clear, specific, and self-consistent?",
    scale=(
        ScoreLevel(
            score=0, 
            label="bad",
            description="Ambiguous/contradictory/missing key details even after context; clarification required."
        ),
        ScoreLevel(
            score=1, 
            label="partial", 
            description="Minor ambiguity; 1–2 missing details; can proceed with minimal clarification."
        ),
        ScoreLevel(
            score=2, 
            label="good",
            description="Clear, unambiguous; enough detail to proceed without assumptions."
        ),
    ),
    weight=1.0,
)

# =============================================================================
# KB/ENUM SELECTION DIMENSIONS
# =============================================================================

TOPIC_RELEVANCE = Dimension(
    id="topic_relevance",
    name="Topic Relevance", 
    description="Does the selected ENUM actually match the user's intent, as expressed in the contexted query and supported by dialog?",
    scale=(
        ScoreLevel(
            score=0, 
            label="bad", 
            description="ENUM is clearly wrong, too generic, or unrelated to user intent."
        ),
        ScoreLevel(
            score=1, 
            label="partial", 
            description="ENUM is adjacent but not exact, or user intent spans multiple ENUMs."
        ),
        ScoreLevel(
            score=2, 
            label="good", 
            description="ENUM directly addresses the query topic/intent; name and description cover exactly what user is asking."
        ),
    ),
    weight=1.5,
)


INFORMATION_COMPLETENESS = Dimension(
    id="information_completeness",
    name="Information Completeness",
    description="Given this ENUM's description (and optional KB retrievals), should the bot have been able to answer without fallback?",
    scale=(
        ScoreLevel(
            score=0, 
            label="bad", 
            description="ENUM description lacks critical information (too shallow or off-scope); fallback was justified."
        ),
        ScoreLevel(
            score=1, 
            label="partial", 
            description="ENUM has partial info (gap exists), but at least a partial/safe answer was possible; fallback was premature."
        ),
        ScoreLevel(
            score=2, 
            label="good", 
            description="ENUM description contains enough info to answer the exact user question; fallback was NOT justified."
        ),
    ),
    weight=1.5,
)

# =============================================================================
# OUTPUT-PROMPT / MODEL-COMPARISON DIMENSIONS
# =============================================================================

FACTUAL_COMPLETENESS = Dimension(
    id="factual_completeness",
    name="Factual Completeness",
    description=(
        "Are all key facts from the baseline response preserved in the "
        "challenger response?  Cross-check against ENUM descriptions."
    ),
    scale=(
        ScoreLevel(
            score=0, label="poor",
            description=(
                "Important facts from the baseline are missing, or the "
                "response is substantially incomplete."
            ),
        ),
        ScoreLevel(
            score=1, label="acceptable",
            description=(
                "One or two secondary facts are missing but the core "
                "answer is intact."
            ),
        ),
        ScoreLevel(
            score=2, label="good",
            description=(
                "All key facts from the baseline are preserved; no "
                "meaningful information is lost."
            ),
        ),
    ),
    weight=2.0,
)

HALLUCINATION_CONTROL = Dimension(
    id="hallucination_control",
    name="Hallucination Control",
    description=(
        "Does the challenger response avoid introducing claims not "
        "grounded in the ENUM descriptions?"
    ),
    scale=(
        ScoreLevel(
            score=0, label="poor",
            description=(
                "Contains one or more clearly fabricated or contradictory "
                "claims relative to the ENUM content."
            ),
        ),
        ScoreLevel(
            score=1, label="acceptable",
            description=(
                "Contains one minor ungrounded claim that does not "
                "materially mislead the user."
            ),
        ),
        ScoreLevel(
            score=2, label="good",
            description=(
                "Every claim is grounded in the ENUM descriptions or is "
                "a reasonable inference from them."
            ),
        ),
    ),
    weight=2.0,
)

CLARITY = Dimension(
    id="clarity",
    name="Clarity",
    description=(
        "Is the response easy for a Czech-speaking bank "
        "customer to understand?"
    ),
    scale=(
        ScoreLevel(
            score=0, label="poor",
            description="Confusing or ambiguous wording that would likely mislead the customer.",
        ),
        ScoreLevel(
            score=1, label="acceptable",
            description="Mostly clear, but one passage could be misunderstood.",
        ),
        ScoreLevel(
            score=2, label="good",
            description="Clear, unambiguous language; no confusing phrasing.",
        ),
    ),
    weight=1.0,
)

STRUCTURE = Dimension(
    id="structure",
    name="Structure",
    description=(
        "Is the response logically organized (headings, "
        "lists, flow)?"
    ),
    scale=(
        ScoreLevel(
            score=0, label="poor",
            description="Disorganized; key information is buried or hard to find.",
        ),
        ScoreLevel(
            score=1, label="acceptable",
            description=(
                "Acceptable organization with minor issues (e.g. one "
                "long paragraph where a list would help)."
            ),
        ),
        ScoreLevel(
            score=2, label="good",
            description="Well organized; information is easy to scan and follow.",
        ),
    ),
    weight=1.0,
)

CONCISENESS = Dimension(
    id="conciseness",
    name="Conciseness",
    description=(
        "Is the response appropriately concise without "
        "losing important detail?"
    ),
    scale=(
        ScoreLevel(
            score=0, label="poor",
            description=(
                "Excessively verbose (adds significant noise) or overly "
                "terse (omits needed context)."
            ),
        ),
        ScoreLevel(
            score=1, label="acceptable",
            description="Slightly too verbose or slightly too sparse, but overall acceptable.",
        ),
        ScoreLevel(
            score=2, label="good",
            description="Right level of detail; no filler or unnecessary repetition.",
        ),
    ),
    weight=1.0,
)

TONE_APPROPRIATENESS = Dimension(
    id="tone_appropriateness",
    name="Tone Appropriateness",
    description=(
        "Is the tone professional and appropriate for banking customer "
        "service in Czech?"
    ),
    scale=(
        ScoreLevel(
            score=0, label="poor",
            description=(
                "Clearly inappropriate tone (condescending, overly "
                "informal, aggressive, or insensitive)."
            ),
        ),
        ScoreLevel(
            score=1, label="acceptable",
            description=(
                "Mostly appropriate with one minor tone slip (too casual, "
                "too formal, or slightly robotic)."
            ),
        ),
        ScoreLevel(
            score=2, label="good",
            description=(
                "Professional, friendly, and helpful; consistent with "
                "banking standards."
            ),
        ),
    ),
    weight=1.0,
)

# =============================================================================
# DIMENSION REGISTRY
# =============================================================================

# External dimensions registered at runtime by users of the library.
# This allows projects that install hg-ds-evals to define and register
# their own dimensions without modifying the library source.
_DIMENSION_REGISTRY: dict[str, Dimension] = {}


def register_dimension(dimension: Dimension) -> None:
    """
    Register a custom dimension so it is discoverable by the catalog.

    Once registered, the dimension can be looked up via ``get_dimension_by_id``
    and will appear in ``list_all_dimensions``.  It can also be referenced by
    ID in experiment YAML configs.

    Args:
        dimension: The Dimension object to register.

    Example::

        from hg_ds_evals.core.types import Dimension, ScoreLevel
        from hg_ds_evals.rubrics.dimensions.catalog import register_dimension

        RESPONSE_TONE = Dimension(
            id="response_tone",
            name="Response Tone",
            description="Is the tone professional and empathetic?",
            scale=(
                ScoreLevel(0, "bad", "Rude or dismissive"),
                ScoreLevel(1, "partial", "Neutral but lacks empathy"),
                ScoreLevel(2, "good", "Professional and empathetic"),
            ),
            weight=1.0,
        )
        register_dimension(RESPONSE_TONE)
    """
    _DIMENSION_REGISTRY[dimension.id] = dimension


def register_dimensions(*dimensions: Dimension) -> None:
    """
    Register multiple custom dimensions at once.

    Args:
        *dimensions: Dimension objects to register.

    Example::

        register_dimensions(RESPONSE_TONE, GRAMMAR_QUALITY, SAFETY_CHECK)
    """
    for dim in dimensions:
        register_dimension(dim)


def unregister_dimension(dim_id: str) -> None:
    """
    Remove a previously registered dimension.

    Args:
        dim_id: The ID of the dimension to remove.

    Raises:
        KeyError: If the dimension ID is not in the registry.
    """
    if dim_id not in _DIMENSION_REGISTRY:
        raise KeyError(
            f"Dimension '{dim_id}' is not in the registry. "
            f"Registered: {sorted(_DIMENSION_REGISTRY.keys())}"
        )
    del _DIMENSION_REGISTRY[dim_id]


def list_registered_dimensions() -> list[Dimension]:
    """Return only the externally registered dimensions."""
    return list(_DIMENSION_REGISTRY.values())


# =============================================================================
# COLLECT ALL DEFINED DIMENSIONS (built-in + registered)
# =============================================================================

def _collect_builtin_dimensions() -> list[Dimension]:
    """Collect all Dimension instances defined as module-level constants."""
    import sys
    current_module = sys.modules[__name__]
    dimensions = []
    for name in dir(current_module):
        if name.isupper() and not name.startswith("_"):
            obj = getattr(current_module, name)
            # Use class name check to handle module reloading scenarios
            # else the tests fail due to multiple Dimension class copies
            if type(obj).__name__ == "Dimension":
                dimensions.append(obj)
    return dimensions


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def list_all_dimensions() -> list[Dimension]:
    """
    Return all available dimensions (built-in + registered).

    Built-in dimensions are the constants defined in this module.
    Registered dimensions are added at runtime via ``register_dimension``.
    If a registered dimension has the same ID as a built-in one, the
    registered version takes precedence.
    """
    # Start with built-ins keyed by ID
    by_id = {d.id: d for d in _collect_builtin_dimensions()}
    # Registered dimensions override built-ins with the same ID
    by_id.update(_DIMENSION_REGISTRY)
    return list(by_id.values())


def get_dimension_by_id(dim_id: str) -> Dimension | None:
    """
    Get a dimension by its ID (checks registry first, then built-ins).

    Args:
        dim_id: The dimension ID to look up.

    Returns:
        The Dimension if found, None otherwise.
    """
    # Registry takes precedence (allows overriding built-ins)
    if dim_id in _DIMENSION_REGISTRY:
        return _DIMENSION_REGISTRY[dim_id]
    for dim in _collect_builtin_dimensions():
        if dim.id == dim_id:
            return dim
    return None


# =============================================================================
# YAML-BASED PERSISTENT DIMENSIONS
# =============================================================================

def load_dimensions_from_yaml(yaml_path: str | Path) -> list[Dimension]:
    """
    Load dimensions from a YAML file and register them in the catalog.

    The YAML file should contain a list of dimension definitions under a
    top-level ``dimensions`` key.  Each dimension needs at minimum an ``id``
    and a ``scale`` list.

    Loaded dimensions are automatically registered so they are available
    via ``get_dimension_by_id`` and ``list_all_dimensions``.

    Args:
        yaml_path: Path to the YAML file with dimension definitions.

    Returns:
        List of Dimension objects that were loaded and registered.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        KeyError: If a dimension entry is missing required fields.

    YAML format example::

        dimensions:
          - id: response_tone
            name: Response Tone
            description: "Is the tone professional and empathetic?"
            weight: 1.0
            scale:
              - score: 0
                label: bad
                description: "Rude or dismissive"
              - score: 1
                label: partial
                description: "Neutral but lacks empathy"
              - score: 2
                label: good
                description: "Professional and empathetic"

          - id: grammar_quality
            name: Grammar Quality
            description: "Is the response grammatically correct?"
            weight: 0.5
            scale:
              - score: 0
                label: bad
                description: "Multiple errors"
              - score: 1
                label: partial
                description: "Minor errors"
              - score: 2
                label: good
                description: "No errors"

    Usage::

        from hg_ds_evals.rubrics.dimensions.catalog import load_dimensions_from_yaml

        # Load once at startup — dimensions persist for the session
        dims = load_dimensions_from_yaml("my_project/custom_dimensions.yaml")

        # Now usable everywhere
        get_dimension_by_id("response_tone")  # returns the loaded Dimension
    """
    import yaml

    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dimensions YAML not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    dims_data = data.get("dimensions", [])
    loaded: list[Dimension] = []

    for entry in dims_data:
        scale = tuple(
            ScoreLevel(
                score=s["score"],
                label=s["label"],
                description=s["description"],
            )
            for s in entry.get("scale", [])
        )
        dim = Dimension(
            id=entry["id"],
            name=entry.get("name", entry["id"]),
            description=entry.get("description", ""),
            scale=scale,
            weight=entry.get("weight", 1.0),
        )
        register_dimension(dim)
        loaded.append(dim)

    return loaded