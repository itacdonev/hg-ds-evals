# hg_ds_evals/rubrics/loader.py
"""
YAML-driven experiment loader for building Rubrics from experiment config files.

This module bridges experiment YAML files and the Rubric/PromptBuilder system.
Users define a single YAML file per evaluation experiment containing rubric
configuration, model settings, dataset info, and paths. This module parses the
rubric section and builds a fully configured Rubric object.

Design:
    - YAML `rubric.base` references a pre-defined Python rubric (e.g., "fallback")
    - Sections like `input_fields`, `output_schema`, `dimensions` override the base
    - Omitting a section keeps the base defaults
    - Setting `base: null` builds a rubric entirely from YAML (no Python defaults)

Usage:
    from hg_ds_evals.rubrics.loader import load_experiment_config, build_rubric_from_config

    # Load full experiment config
    config = load_experiment_config("experiments/fallback_exp_001_baseline.yaml")

    # Build rubric from the config's rubric section
    rubric = build_rubric_from_config(config)

    # Or one-liner
    rubric = load_experiment_rubric("experiments/fallback_exp_001_baseline.yaml")
"""

from pathlib import Path
from typing import Any, Optional

import yaml

from hg_ds_evals.core.types import (
    Dimension,
    InputField,
    OutputField,
    OutputSchema,
    ScoreLevel,
)
from hg_ds_evals.rubrics.base import Rubric, RubricMetadata


# =============================================================================
# BASE RUBRIC REGISTRY
# =============================================================================

# Lazy-loaded registry mapping base names to rubric objects.
# We use a function to avoid circular imports at module level.
# When _get_base_registry() is first called, it checks if _BASE_REGISTRY is None, 
# and if so, imports and registers the predefined rubrics (like FALLBACK_RUBRIC). 
# This lazy-loading pattern avoids circular import issues since the rubric 
# definitions themselves might import from this module.

_BASE_REGISTRY: dict[str, Rubric] | None = None


def _get_base_registry() -> dict[str, Rubric]:
    """
    Build and cache the registry of base rubrics.
    
    Maps short names (used in YAML `rubric.base`) to pre-defined Rubric objects.
    Add new base rubrics here as they are created.
    """
    global _BASE_REGISTRY
    if _BASE_REGISTRY is None:
        from hg_ds_evals.rubrics.fallback import FALLBACK_RUBRIC, FALLBACK_RUBRIC_FULL
        from hg_ds_evals.rubrics.output_prompt import OUTPUT_PROMPT_RUBRIC
        _BASE_REGISTRY = {
            "fallback": FALLBACK_RUBRIC,
            "fallback_full": FALLBACK_RUBRIC_FULL,
            "output_prompt": OUTPUT_PROMPT_RUBRIC,
        }
    return _BASE_REGISTRY


def register_base_rubric(name: str, rubric: Rubric) -> None:
    """
    Register a new base rubric that can be referenced in YAML configs.
    The register_base_rubric function allows you to dynamically add 
    new base rubrics to the registry at runtime. It provides an extensibility 
    point so users can register custom rubrics without modifying this file. 
    Once registered, these rubrics can be referenced by name in YAML config 
    files using the rubric.base field.
    
    Args:
        name: Short name to use in YAML `rubric.base` field.
        rubric: The Rubric object to register.
    
    Example:
        from hg_ds_evals.rubrics.loader import Rubric, register_base_rubric

        # Define a custom rubric
        my_rubric = Rubric(
            metadata=RubricMetadata(id="custom", name="My Custom Rubric"),
            dimensions=(...)
        )

        # Register it so YAML configs can use `base: "my_custom"`
        register_base_rubric("my_custom", my_rubric)
    """
    registry = _get_base_registry()
    registry[name] = rubric


# =============================================================================
# YAML PARSING HELPERS
# =============================================================================

def _parse_score_level(data: dict) -> ScoreLevel:
    """Parse a ScoreLevel from a YAML dict."""
    return ScoreLevel(
        score=data["score"],
        label=data["label"],
        description=data["description"],
    )


def _parse_dimension(data: dict, base_dim: Optional[Dimension] = None) -> Dimension:
    """
    Parse a Dimension from a YAML dict, optionally merging with a base dimension.
    
    If base_dim is provided, only the fields present in data override the base.
    If base_dim is None, all fields must be present in data.
    """
    if base_dim is not None:
        # Merge: YAML overrides base
        scale = base_dim.scale
        if "scale" in data:
            scale = tuple(_parse_score_level(s) for s in data["scale"])
        
        return Dimension(
            id=data.get("id", base_dim.id),
            name=data.get("name", base_dim.name),
            description=data.get("description", base_dim.description),
            scale=scale,
            weight=data.get("weight", base_dim.weight),
        )
    else:
        # Full definition from YAML
        scale = tuple(_parse_score_level(s) for s in data.get("scale", []))
        return Dimension(
            id=data["id"],
            name=data.get("name", data["id"]),
            description=data.get("description", ""),
            scale=scale,
            weight=data.get("weight", 1.0),
        )


def _parse_input_fields(fields_data: list[dict]) -> tuple[InputField, ...]:
    """Parse a tuple of InputField from YAML list."""
    return tuple(
        InputField(
            name=f["name"],
            description=f.get("description", ""),
            required=f.get("required", True),
        )
        for f in fields_data
    )


def _parse_output_schema(schema_data: dict) -> OutputSchema:
    """Parse an OutputSchema from a YAML dict."""
    fields = tuple(
        OutputField(
            name=f["name"],
            field_type=f.get("type", "string"),
            description=f.get("description", ""),
            enum=tuple(f["enum"]) if "enum" in f else (),
        )
        for f in schema_data.get("fields", [])
    )
    return OutputSchema(
        fields=fields,
        additional_instructions=schema_data.get("additional_instructions", ""),
    )


# =============================================================================
# MAIN BUILDER FUNCTIONS
# =============================================================================

def load_experiment_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load and return the full experiment configuration from a YAML file.
    
    Args:
        config_path: Path to the experiment YAML file.
        
    Returns:
        Dictionary with the full experiment configuration.
        
    Raises:
        FileNotFoundError: If the YAML file does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Inject experiment name from filename (without extension)
    config.setdefault("experiment", {})
    config["experiment"]["name"] = config_path.stem
    config["experiment"]["config_path"] = str(config_path)
    
    return config


def build_rubric_from_config(config: dict[str, Any]) -> Rubric:
    """
    Build a Rubric object from an experiment configuration dict.
    
    The config should have a `rubric` section with optional `base`, `dimensions`,
    `input_fields`, `output_schema`, `pass_threshold`, and `judge_instructions`.
    
    Resolution order:
        1. Load base rubric (if `rubric.base` is set)
        2. Apply dimension overrides/filters
        3. Apply input_fields override (full replacement if present)
        4. Apply output_schema override (full replacement if present)
        5. Apply pass_threshold and judge_instructions overrides
    
    Args:
        config: Full experiment configuration dict (from load_experiment_config).
        
    Returns:
        Fully configured Rubric instance.
    """
    rubric_config = config.get("rubric", {})
    experiment_config = config.get("experiment", {})
    
    # ── Step 1: Resolve base rubric ──────────────────────────────
    base_name = rubric_config.get("base")
    
    if base_name:
        registry = _get_base_registry()
        if base_name not in registry:
            available = ", ".join(sorted(registry.keys()))
            raise ValueError(
                f"Unknown base rubric '{base_name}'. Available: {available}"
            )
        rubric = registry[base_name]
    else:
        # No base - start with empty rubric
        rubric = Rubric(
            metadata=RubricMetadata(
                id="custom",
                name="Custom Rubric",
                version="1.0.0",
            ),
            dimensions=(),
        )
    
    # ── Step 2: Build metadata from experiment + base ────────────
    metadata = RubricMetadata(
        id=experiment_config.get("name", rubric.metadata.id),
        name=rubric.metadata.name,
        version=experiment_config.get("version", rubric.metadata.version),
        description=experiment_config.get("description", rubric.metadata.description),
        persona=experiment_config.get("persona", rubric.metadata.persona),
        author=experiment_config.get("author", rubric.metadata.author),
    )
    
    # ── Step 3: Apply dimension overrides/filters ────────────────
    dims_config = rubric_config.get("dimensions")
    
    if dims_config is not None:
        # YAML specifies dimensions → use only these (filter + override)
        new_dimensions = []
        
        # Also check catalog for dimensions not in base
        from hg_ds_evals.rubrics.dimensions.catalog import get_dimension_by_id
        
        for dim_data in dims_config:
            dim_id = dim_data["id"]
            
            # Try to find in base rubric first, then catalog
            base_dim = rubric.get_dimension(dim_id)
            if base_dim is None:
                base_dim = get_dimension_by_id(dim_id)
            
            if base_dim is not None:
                # Merge overrides onto base dimension
                new_dimensions.append(_parse_dimension(dim_data, base_dim))
            else:
                # Fully new dimension defined in YAML
                new_dimensions.append(_parse_dimension(dim_data))
        
        dimensions = tuple(new_dimensions)
    else:
        # No dimensions in YAML → keep base defaults
        dimensions = rubric.dimensions
    
    # ── Step 4: Apply input_fields override ──────────────────────
    input_fields_config = rubric_config.get("input_fields")
    
    if input_fields_config is not None:
        input_fields = _parse_input_fields(input_fields_config)
    else:
        input_fields = rubric.input_fields
    
    # ── Step 5: Apply output_schema override ─────────────────────
    output_schema_config = rubric_config.get("output_schema")
    
    if output_schema_config is not None:
        output_schema = _parse_output_schema(output_schema_config)
    else:
        output_schema = rubric.output_schema
    
    # ── Step 6: Apply scalar overrides ───────────────────────────
    pass_threshold = rubric_config.get("pass_threshold", rubric.pass_threshold)
    judge_instructions = rubric_config.get("judge_instructions", rubric.judge_instructions)
    
    # ── Step 7: Apply prompt-section overrides ───────────────────
    critical_evaluation_rules = rubric_config.get(
        "critical_evaluation_rules", rubric.critical_evaluation_rules
    )
    system_context = rubric_config.get("system_context", rubric.system_context)
    root_cause_categories = rubric_config.get(
        "root_cause_categories", rubric.root_cause_categories
    )
    domain_specific_guidance = rubric_config.get(
        "domain_specific_guidance", rubric.domain_specific_guidance
    )
    final_reminders = rubric_config.get("final_reminders", rubric.final_reminders)
    
    # ── Build final rubric ───────────────────────────────────────
    return Rubric(
        metadata=metadata,
        dimensions=dimensions,
        input_fields=input_fields,
        output_schema=output_schema,
        pass_threshold=pass_threshold,
        judge_instructions=judge_instructions,
        critical_evaluation_rules=critical_evaluation_rules,
        system_context=system_context,
        root_cause_categories=root_cause_categories,
        domain_specific_guidance=domain_specific_guidance,
        final_reminders=final_reminders,
    )


def load_experiment_rubric(config_path: str | Path) -> Rubric:
    """
    Load experiment YAML and build a Rubric from it.
    
    Args:
        config_path: Path to the experiment YAML file.
        
    Returns:
        Fully configured Rubric instance.
        
    Example:
        rubric = load_experiment_rubric("experiments/fallback_exp_001_baseline.yaml")
        builder = PromptBuilder(rubric=rubric)
    """
    config = load_experiment_config(config_path)
    return build_rubric_from_config(config)


def get_experiment_name(config_path: str | Path) -> str:
    """
    Extract experiment name from YAML filename (stem without extension).
    This name is used for checkpoint files and DBX result table naming.
    
    Args:
        config_path: Path to the experiment YAML file.
        
    Returns:
        Experiment name string (e.g., "fallback_exp_001_baseline").
    """
    return Path(config_path).stem
