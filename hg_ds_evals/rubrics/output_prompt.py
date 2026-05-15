# hg_ds_evals/rubrics/output_prompt.py
"""
Output-prompt evaluation rubric for Hey George model comparison.

This module provides pre-composed rubrics for evaluating output-prompt
responses in a comparative setting (e.g., GPT-4o baseline vs GPT-4.1
challenger).  Dimensions are imported from the shared catalog.

Usage:
    from hg_ds_evals.rubrics.output_prompt import OUTPUT_PROMPT_RUBRIC

    rubric = OUTPUT_PROMPT_RUBRIC
    print(rubric.describe())
"""

from hg_ds_evals.rubrics.base import Rubric, RubricMetadata
from hg_ds_evals.rubrics.dimensions.catalog import *
from hg_ds_evals.core.types import InputField, OutputField, OutputSchema


# =============================================================================
# DIMENSION SETS
# =============================================================================

OUTPUT_PROMPT_DIMENSIONS_CORRECTNESS = (
    FACTUAL_COMPLETENESS,
    HALLUCINATION_CONTROL,
)

OUTPUT_PROMPT_DIMENSIONS_LANGUAGE = (
    CLARITY,
    STRUCTURE,
    CONCISENESS,
    TONE_APPROPRIATENESS,
)

OUTPUT_PROMPT_DIMENSIONS_ALL = (
    *OUTPUT_PROMPT_DIMENSIONS_CORRECTNESS,
    *OUTPUT_PROMPT_DIMENSIONS_LANGUAGE,
)


# =============================================================================
# INPUT FIELDS FOR OUTPUT-PROMPT EVALUATION
# =============================================================================

OUTPUT_PROMPT_INPUT_FIELDS = (
    InputField(
        name="input",
        description="The exact user message (in Czech) that triggered the bot response.",
        required=True,
    ),
    InputField(
        name="run_enum_gpt4o",
        description=(
            "Comma-separated Phase II ENUM(s) selected by the bot for this query. "
            "Identical across models."
        ),
        required=True,
    ),
    InputField(
        name="run_enum_desc_cz",
        description=(
            "Full Czech descriptions of the selected ENUM(s) from the knowledge base. "
            "This is the factual ground truth."
        ),
        required=True,
    ),
    InputField(
        name="run_enum_desc_en",
        description=(
            "Full English descriptions of the selected ENUM(s) from the knowledge base."
        ),
        required=True,
    ),
    InputField(
        name="run_status_gpt4o",
        description=(
            'Bot status for this query: "continue", "partialFallback", '
            '"outputFallback", or "disambiguation". Identical across models.'
        ),
        required=True,
    ),
    InputField(
        name="run_response_gpt4o",
        description="The GPT-4o (baseline) response. Reference for comparison.",
        required=True,
    ),
    InputField(
        name="run_response_gpt41",
        description="The GPT-4.1 (challenger) response. Primary target of evaluation.",
        required=True,
    ),
)


# =============================================================================
# OUTPUT SCHEMA FOR OUTPUT-PROMPT EVALUATION
# =============================================================================

OUTPUT_PROMPT_OUTPUT_SCHEMA = OutputSchema(
    fields=(
        OutputField(
            name="comparison_verdict",
            field_type="string",
            description=(
                "Overall quality verdict for the challenger response "
                "relative to the baseline."
            ),
            enum=("better", "equivalent", "worse", "mixed"),
        ),
        OutputField(
            name="missing_facts",
            field_type="array",
            description=(
                "List of facts present in the baseline (and grounded in "
                "ENUM descriptions) that are absent from the challenger."
            ),
        ),
        OutputField(
            name="hallucinated_claims",
            field_type="array",
            description=(
                "List of claims in the challenger NOT supported by "
                "the ENUM descriptions."
            ),
        ),
        OutputField(
            name="added_information",
            field_type="array",
            description=(
                "List of facts in the challenger NOT in the baseline "
                "but ARE grounded in the ENUM descriptions."
            ),
        ),
        OutputField(
            name="language_quality_verdict",
            field_type="string",
            description=(
                "Summary comparison of language quality between "
                "challenger and baseline."
            ),
            enum=("better", "equivalent", "worse", "mixed"),
        ),
        OutputField(
            name="tone_issues",
            field_type="array",
            description="Specific tone problems in the challenger response.",
        ),
        OutputField(
            name="baseline_has_errors",
            field_type="boolean",
            description=(
                "True if the baseline response contains factual errors "
                "or hallucinations relative to the ENUM descriptions."
            ),
        ),
        OutputField(
            name="baseline_error_details",
            field_type="string",
            description="Brief description of baseline errors (empty if none).",
        ),
        OutputField(
            name="fallback_quality",
            field_type="string",
            description=(
                "Rates the challenger's fallback handling (only when "
                "status is outputFallback or partialFallback)."
            ),
            enum=(
                "better_than_baseline",
                "equivalent_to_baseline",
                "worse_than_baseline",
                "not_applicable",
            ),
        ),
        OutputField(
            name="overall_explanation",
            field_type="string",
            description=(
                "3-5 sentence summary of the comparison. Must reference "
                "rubric scores and justify comparison_verdict."
            ),
        ),
        OutputField(
            name="upgrade_safe",
            field_type="boolean",
            description=(
                "True if switching from baseline to challenger is safe "
                "for this test case."
            ),
        ),
    ),
    additional_instructions=(
        "Set upgrade_safe=false if factual_completeness=0 OR hallucination_control=0.\n"
        "Set upgrade_safe=false if the weighted average of all dimensions < 1.4.\n"
        'Set fallback_quality="not_applicable" when run_status is "continue" or "disambiguation".\n'
        "Keep overall_explanation between 150 and 500 characters.\n"
        "All array fields must be arrays of strings. Use [] when nothing applies.\n"
        'If baseline_has_errors=false, set baseline_error_details to "".'
    ),
)


# =============================================================================
# PRE-COMPOSED RUBRIC
# =============================================================================

OUTPUT_PROMPT_RUBRIC = Rubric(
    metadata=RubricMetadata(
        id="output_prompt_eval",
        name="Output-Prompt Evaluator",
        version="1.0.0",
        description=(
            "Comparative evaluation of output-prompt responses. "
            "Evaluates correctness (factual completeness, hallucination control) "
            "and language quality (clarity, structure, conciseness, tone) of a "
            "challenger model against a baseline."
        ),
        author="DS Team",
    ),
    dimensions=OUTPUT_PROMPT_DIMENSIONS_ALL,
    input_fields=OUTPUT_PROMPT_INPUT_FIELDS,
    output_schema=OUTPUT_PROMPT_OUTPUT_SCHEMA,
    pass_threshold=1.4,
    judge_instructions="",
)
