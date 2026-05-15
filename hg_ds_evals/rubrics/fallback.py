# hg_ds_evals/rubrics/fallback.py
"""
Fallback evaluation rubric for Hey George KB/ENUM selection.

This module provides pre-composed rubrics for evaluating chatbot fallback
scenarios - cases where the bot said "I don't have that information" when
it potentially could have answered from the knowledge base.

Usage:
    from hg_ds_evals.rubrics.fallback import FALLBACK_RUBRIC, create_fallback_rubric
    
    # Use pre-defined rubric directly
    rubric = FALLBACK_RUBRIC
    print(rubric.describe())
    
    # Or customize via factory function
    rubric = create_fallback_rubric(exclude_dimensions=["api_vs_kb_choice"])
"""

from pathlib import Path
from hg_ds_evals.rubrics.base import Rubric, RubricMetadata
from hg_ds_evals.rubrics.dimensions.catalog import *
from hg_ds_evals.core.types import InputField, OutputSchema, OutputField


# =============================================================================
# INPUT FIELDS FOR FALLBACK EVALUATION
# =============================================================================

FALLBACK_INPUT_FIELDS = (
    InputField(
        name="user_query",
        description="The exact user message that triggered the fallback.",
        required=True,
    ),
    InputField(
        name="user_query_contexted",
        description="The user message corrected/reformulated using previous dialog context.",
        required=True,
    ),
    InputField(
        name="conversation_history",
        description="Previous dialog between user and bot (for context).",
        required=False,
    ),
    InputField(
        name="selected_enum_name",
        description="The ENUM(s) the bot selected to answer the query.",
        required=True,
    ),
    InputField(
        name="selected_enum_description",
        description="Full description/definition of the selected ENUM(s).",
        required=True,
    ),
    InputField(
        name="bot_answer",
        description="The actual fallback response sent to the user.",
        required=True,
    ),
)


# =============================================================================
# OUTPUT SCHEMA FOR FALLBACK EVALUATION
# =============================================================================

FALLBACK_OUTPUT_SCHEMA = OutputSchema(
    fields=(
        OutputField(
            name="fallback_justified",
            field_type="boolean",
            description="True if fallback was appropriate; false if bot should have answered from ENUM.",
        ),
        OutputField(
            name="root_cause",
            field_type="string",
            description="Primary reason for fallback (only if fallback_justified=true).",
            enum=(
                "wrong_enum_selected",
                "enum_insufficient_depth",
                "dialog_misalignment",
                "should_call_api_not_kb",
                "out_of_scope_or_true_missing_info",
                "unclear",
            ),
        ),
        OutputField(
            name="critical_gaps",
            field_type="array",
            description="List of missing information or gaps that led to fallback.",
        ),
        OutputField(
            name="overall_explanation",
            field_type="string",
            description="2-4 sentence summary consistent with rubric scores and root cause.",
        ),
        OutputField(
            name="expected_answer",
            field_type="string",
            description="Best possible answer given available information.",
        ),
    ),
    additional_instructions=(
        "Set fallback_justified=false if information_completeness=2 AND topic_relevance=2.\n"
        "Keep overall_explanation under 300 characters."
    ),
)

# =============================================================================
# DIMENSION SETS
# Can define multiple rubrics sets here for different use cases
# =============================================================================

# Core dimensions for fallback evaluation (minimal set)
FALLBACK_DIMENSIONS_CORE = (
    USER_QUERY_CLARITY,
    TOPIC_RELEVANCE,
)

# All dimensions for fallback evaluation (full set)
FALLBACK_DIMENSIONS_FULL = (
    USER_QUERY_CLARITY,
    TOPIC_RELEVANCE,
    INFORMATION_COMPLETENESS,
)

# =============================================================================
# PRE-COMPOSED RUBRICS
# Can define multiple rubrics here for different use cases
# =============================================================================

FALLBACK_RUBRIC = Rubric(
    metadata=RubricMetadata(
        id="fallback_eval",
        name="Fallback Evaluator",
        version="1.0.0",
        description=(
            "Evaluates KB ENUM sufficiency for user queries in fallback scenarios. "
            "Assesses whether the bot's knowledge base selection was sufficient to "
            "answer the user query without falling back."
        ),
        author="DS Team",
    ),
    dimensions=FALLBACK_DIMENSIONS_CORE,
    input_fields=FALLBACK_INPUT_FIELDS,
    output_schema=FALLBACK_OUTPUT_SCHEMA,
    pass_threshold=1.5,
    judge_instructions="",
)

FALLBACK_RUBRIC_FULL = Rubric(
    metadata=RubricMetadata(
        id="fallback_eval_full",
        name="Fallback Evaluator (Full)",
        version="1.0.0",
        description=(
            "Comprehensive fallback evaluation with all 7 dimensions including "
            "dialog awareness, actionability, and API vs KB choice assessment."
        ),
        author="DS Team",
    ),
    dimensions=FALLBACK_DIMENSIONS_FULL,
    input_fields=FALLBACK_INPUT_FIELDS,
    output_schema=FALLBACK_OUTPUT_SCHEMA,
    pass_threshold=1.5,
    judge_instructions="",
)

# =============================================================================

def create_fallback_rubric(
    config_override: str | Path = None,
    exclude_dimensions: list[str] = None,
    include_all_dimensions: bool = False,
    pass_threshold: float = None,
    judge_instructions: str = None,
) -> Rubric:
    """
    Create a fallback evaluation rubric with optional customization.
    
    Args:
        config_override: Path to YAML file for configuration overrides.
        exclude_dimensions: List of dimension IDs to exclude from the rubric.
        include_all_dimensions: If True, start with full dimension set (7 dims).
                               If False (default), start with core set (3 dims).
        pass_threshold: Override the default pass threshold.
        judge_instructions: Additional instructions for the LLM judge.
        
    Returns:
        Configured Rubric instance.
        
    Examples:
        # Default rubric (core dimensions)
        rubric = create_fallback_rubric()
        
        # Full rubric with all dimensions
        rubric = create_fallback_rubric(include_all_dimensions=True)
        
        # Exclude specific dimension
        rubric = create_fallback_rubric(
            include_all_dimensions=True,
            exclude_dimensions=["api_vs_kb_choice"]
        )
        
        # With YAML overrides
        rubric = create_fallback_rubric(config_override="configs/strict.yaml")
        
        # Stricter threshold
        rubric = create_fallback_rubric(pass_threshold=1.7)
    """
    # Start with appropriate base rubric
    rubric = FALLBACK_RUBRIC_FULL if include_all_dimensions else FALLBACK_RUBRIC
    
    # Apply exclusions
    if exclude_dimensions:
        for dim_id in exclude_dimensions:
            rubric = rubric.remove_dimension(dim_id)
    
    # Apply threshold override
    if pass_threshold is not None or judge_instructions is not None:
        rubric = Rubric(
            metadata=rubric.metadata,
            dimensions=rubric.dimensions,
            input_fields=rubric.input_fields,
            output_schema=rubric.output_schema,
            pass_threshold=pass_threshold if pass_threshold is not None else rubric.pass_threshold,
            judge_instructions=judge_instructions if judge_instructions is not None else rubric.judge_instructions,
        )
    
    # YAML overrides (we apply this last, so they take precedence)
    if config_override:
        rubric = rubric.override_from_yaml(Path(config_override))
    
    return rubric