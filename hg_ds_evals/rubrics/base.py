from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path
import yaml

from hg_ds_evals.core.types import Dimension, InputField, OutputSchema, OutputField, ScoreLevel


@dataclass
class RubricMetadata:
    """Rubric identification and versioning."""
    id: str
    name: str
    version: str
    description: str = ""
    persona: str = ""
    author: str = ""
    

@dataclass
class Rubric:
    """
    A composed evaluation rubric.
    
    Rubrics are compositions of Dimensions with metadata and configuration.
    They define what to evaluate (dimensions), what inputs are needed (input_fields),
    and what output structure is expected (output_schema).
    
    Attributes:
        metadata: Identification and versioning information for the rubric.
        dimensions: Tuple of Dimension objects that define what to evaluate.
        input_fields: Tuple of InputField objects defining user prompt inputs.
        output_schema: OutputSchema defining expected LLM response structure.
        pass_threshold: Minimum average score (0-2) to consider evaluation "passed".
        judge_instructions: Optional free-form instructions appended to LLM prompt.
        critical_evaluation_rules: Optional rules rendered in the system prompt.
        system_context: Optional domain/system context for the LLM judge.
        root_cause_categories: Optional root-cause taxonomy rendered in the prompt.
        domain_specific_guidance: Optional domain-specific evaluation guidance.
        final_reminders: Optional final reminders appended to the system prompt.

    Usage:
        from hg_ds_evals.rubrics.dimensions.catalog import FALLBACK_DIMENSIONS
        
        rubric = Rubric(
            metadata=RubricMetadata(id="fallback", name="Fallback Eval", version="1.0.0"),
            dimensions=FALLBACK_DIMENSIONS,
            input_fields=FALLBACK_INPUT_FIELDS,
        )
    """
    metadata: RubricMetadata
    dimensions: tuple[Dimension, ...]
    input_fields: tuple[InputField, ...] = ()
    output_schema: Optional[OutputSchema] = None
    pass_threshold: float = 1.5
    judge_instructions: str = ""
    critical_evaluation_rules: str = ""
    system_context: str = ""
    root_cause_categories: str = ""
    domain_specific_guidance: str = ""
    final_reminders: str = ""

    def __post_init__(self):
        """Ensure tuples for immutability."""
        if isinstance(self.dimensions, list):
            object.__setattr__(self, 'dimensions', tuple(self.dimensions))
        if isinstance(self.input_fields, list):
            object.__setattr__(self, 'input_fields', tuple(self.input_fields))
    
    @property
    def dimension_ids(self) -> list[str]:
        """List of dimension IDs in this rubric."""
        return [d.id for d in self.dimensions]
    
    @property
    def total_weight(self) -> float:
        """Sum of all dimension weights."""
        return sum(d.weight for d in self.dimensions)
    
    @property
    def input_field_names(self) -> list[str]:
        """List of input field names."""
        return [f.name for f in self.input_fields]

    def get_dimension(self, dim_id: str) -> Optional[Dimension]:
        """
        Get a dimension by its ID.
        
        Args:
            dim_id: The unique identifier of the dimension.
            
        Returns:
            The Dimension object if found, None otherwise.
        """
        for d in self.dimensions:
            if d.id == dim_id:
                return d
        return None
    
    def get_input_field(self, field_name: str) -> Optional[InputField]:
        """
        Get an input field by its name.
        
        Args:
            field_name: The name of the input field.
            
        Returns:
            The InputField object if found, None otherwise.
        """
        for f in self.input_fields:
            if f.name == field_name:
                return f
        return None
    
    def _carry_fields(self, **overrides) -> dict:
        """Helper: return constructor kwargs preserving all fields, with overrides applied."""
        base = dict(
            metadata=self.metadata,
            dimensions=self.dimensions,
            input_fields=self.input_fields,
            output_schema=self.output_schema,
            pass_threshold=self.pass_threshold,
            judge_instructions=self.judge_instructions,
            critical_evaluation_rules=self.critical_evaluation_rules,
            system_context=self.system_context,
            root_cause_categories=self.root_cause_categories,
            domain_specific_guidance=self.domain_specific_guidance,
            final_reminders=self.final_reminders,
        )
        base.update(overrides)
        return base

    def with_dimensions(self, *dimensions: Dimension) -> "Rubric":
        """Return new rubric with replaced dimensions."""
        return Rubric(**self._carry_fields(dimensions=dimensions))
    
    def with_input_fields(self, *input_fields: InputField) -> "Rubric":
        """Return new rubric with replaced input fields."""
        return Rubric(**self._carry_fields(input_fields=input_fields))
    
    def with_output_schema(self, output_schema: OutputSchema) -> "Rubric":
        """Return new rubric with replaced output schema."""
        return Rubric(**self._carry_fields(output_schema=output_schema))
    
    def add_dimension(self, dimension: Dimension) -> "Rubric":
        """Return new rubric with an added dimension."""
        return Rubric(**self._carry_fields(dimensions=self.dimensions + (dimension,)))
    
    def remove_dimension(self, dim_id: str) -> "Rubric":
        """Return new rubric without the specified dimension."""
        return Rubric(**self._carry_fields(
            dimensions=tuple(d for d in self.dimensions if d.id != dim_id)
        ))
    
    @staticmethod
    def _parse_scale_from_yaml(scale_data) -> tuple[ScoreLevel, ...]:
        """
        Parse a scale definition from YAML into ScoreLevel objects.
        
        Accepts a list of dicts with keys: score, label, description.
        
        Args:
            scale_data: List of dicts from YAML.
            
        Returns:
            Tuple of ScoreLevel objects.
        """
        return tuple(
            ScoreLevel(
                score=s["score"],
                label=s["label"],
                description=s["description"],
            )
            for s in scale_data
        )

    def override_from_yaml(self, yaml_path: Path) -> "Rubric":
        """
        Apply YAML overrides and return new rubric.
        
        The YAML file can override metadata fields, dimension weights/scale,
        pass_threshold, judge_instructions, and prompt-section fields.
        New dimensions not present in the base rubric can be defined fully
        in YAML (must include id, name, description, and scale).
        
        Args:
            yaml_path: Path to the YAML override file.
            
        Returns:
            New Rubric instance with overrides applied.
        """
        with open(yaml_path) as f:
            overrides = yaml.safe_load(f)
        
        # Override metadata
        new_metadata = RubricMetadata(
            id=overrides.get("id", self.metadata.id),
            name=overrides.get("name", self.metadata.name),
            version=overrides.get("version", self.metadata.version),
            description=overrides.get("description", self.metadata.description),
            persona=overrides.get("persona", self.metadata.persona),
            author=overrides.get("author", self.metadata.author),
        )
        
        # Override / add dimensions
        yaml_dims = overrides.get("dimensions", [])
        dim_overrides = {d["id"]: d for d in yaml_dims}
        existing_ids = {dim.id for dim in self.dimensions}
        new_dimensions = []
        
        # 1. Process existing base dimensions (override or keep)
        for dim in self.dimensions:
            if dim.id in dim_overrides:
                override = dim_overrides[dim.id]
                # Parse scale if provided, otherwise keep base scale
                scale = dim.scale
                if "scale" in override and isinstance(override["scale"], list):
                    scale = self._parse_scale_from_yaml(override["scale"])
                new_dim = Dimension(
                    id=dim.id,
                    name=override.get("name", dim.name),
                    description=override.get("description", dim.description),
                    scale=scale,
                    weight=override.get("weight", dim.weight),
                )
                new_dimensions.append(new_dim)
            else:
                new_dimensions.append(dim)
        
        # 2. Add completely new dimensions defined only in YAML
        for dim_data in yaml_dims:
            if dim_data["id"] not in existing_ids:
                scale = ()
                if "scale" in dim_data and isinstance(dim_data["scale"], list):
                    scale = self._parse_scale_from_yaml(dim_data["scale"])
                new_dim = Dimension(
                    id=dim_data["id"],
                    name=dim_data.get("name", dim_data["id"]),
                    description=dim_data.get("description", ""),
                    scale=scale,
                    weight=dim_data.get("weight", 1.0),
                )
                new_dimensions.append(new_dim)
        
        return Rubric(
            metadata=new_metadata,
            dimensions=tuple(new_dimensions),
            input_fields=self.input_fields,
            output_schema=self.output_schema,
            pass_threshold=overrides.get("pass_threshold", self.pass_threshold),
            judge_instructions=overrides.get("judge_instructions", self.judge_instructions),
            critical_evaluation_rules=overrides.get("critical_evaluation_rules", self.critical_evaluation_rules),
            system_context=overrides.get("system_context", self.system_context),
            root_cause_categories=overrides.get("root_cause_categories", self.root_cause_categories),
            domain_specific_guidance=overrides.get("domain_specific_guidance", self.domain_specific_guidance),
            final_reminders=overrides.get("final_reminders", self.final_reminders),
        )
    

    # =========================================================================
    # Display & Export Methods
    # =========================================================================

    def describe(self) -> str:
        """
        Return a formatted string describing the rubric and all dimensions.
        
        Returns:
            Human-readable multi-line string with rubric details.
            
        Example:
            print(rubric.describe())
        """
        lines = [
            f"{'='*60}",
            f"Rubric: {self.metadata.name}",
            f"{'='*60}",
            f"ID: {self.metadata.id}",
            f"Version: {self.metadata.version}",
            f"Description: {self.metadata.description or 'N/A'}",
            f"Pass Threshold: {self.pass_threshold}",
            f"Total Dimensions: {len(self.dimensions)}",
            f"Total Weight: {self.total_weight}",
            "",
        ]
        
        if self.critical_evaluation_rules:
            lines.append(f"Critical Evaluation Rules: {self.critical_evaluation_rules}")
            lines.append("")
        
        if self.system_context:
            lines.append(f"System Context: {self.system_context}")
            lines.append("")
        
        if self.root_cause_categories:
            lines.append(f"Root Cause Categories: {self.root_cause_categories}")
            lines.append("")
        
        if self.domain_specific_guidance:
            lines.append(f"Domain Specific Guidance: {self.domain_specific_guidance}")
            lines.append("")
        
        if self.final_reminders:
            lines.append(f"Final Reminders: {self.final_reminders}")
            lines.append("")
        
        if self.input_fields:
            lines.append("Input Fields:")
            for f in self.input_fields:
                req = "required" if f.required else "optional"
                lines.append(f"  - {f.name} ({req}): {f.description}")
            lines.append("")
        
        lines.append("Dimensions:")

        for i, dim in enumerate(self.dimensions, 1):
            lines.append(f"[{i}] {dim.name} (id: {dim.id}, weight: {dim.weight})")
            lines.append(f"    Description: {dim.description}")
            lines.append(f"    Scale:")
            for level in dim.scale:
                lines.append(f"      {level.score} ({level.label}): {level.description}")
            lines.append("")
        
        return "\n".join(lines)
    
    def describe_dimensions(self) -> str:
        """
        Return a concise table-like description of dimensions only.
        
        Returns:
            Formatted string showing dimension IDs, names, and weights.
        """
        lines = [
            f"{'ID':<30} {'Name':<30} {'Weight':>8}",
            f"{'-'*30} {'-'*30} {'-'*8}",
        ]
        for dim in self.dimensions:
            lines.append(f"{dim.id:<30} {dim.name:<30} {dim.weight:>8.1f}")
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Export rubric as a dictionary (for serialization/logging)."""
        result = {
            "metadata": {
                "id": self.metadata.id,
                "name": self.metadata.name,
                "version": self.metadata.version,
                "description": self.metadata.description,
                "persona": self.metadata.persona,
                "author": self.metadata.author,
            },
            "dimensions": [
                {
                    "id": d.id,
                    "name": d.name,
                    "description": d.description,
                    "weight": d.weight,
                    "scale": [
                        {"score": s.score, "label": s.label, "description": s.description} 
                        for s in d.scale
                    ],
                }
                for d in self.dimensions
            ],
            "input_fields": [
                {"name": f.name, "description": f.description, "required": f.required}
                for f in self.input_fields
            ],
            "pass_threshold": self.pass_threshold,
            "judge_instructions": self.judge_instructions,
            "critical_evaluation_rules": self.critical_evaluation_rules,
            "system_context": self.system_context,
            "root_cause_categories": self.root_cause_categories,
            "domain_specific_guidance": self.domain_specific_guidance,
            "final_reminders": self.final_reminders,
        }
        
        if self.output_schema:
            result["output_schema"] = {
                "fields": [
                    {
                        "name": f.name,
                        "type": f.field_type,
                        "description": f.description,
                        "enum": list(f.enum) if f.enum else None,
                    }
                    for f in self.output_schema.fields
                ],
                "additional_instructions": self.output_schema.additional_instructions,
            }
        
        return result
    
    def to_yaml(self) -> str:
        """
        Export rubric as YAML string.
        """
        return yaml.dump(self.to_dict(), sort_keys=False, default_flow_style=False)
    
    def __repr__(self) -> str:
        return (
            f"Rubric(id='{self.metadata.id}', version='{self.metadata.version}', "
            f"dimensions={len(self.dimensions)}, pass_threshold={self.pass_threshold})"
        )