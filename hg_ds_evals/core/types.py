# hg_ds_evals/core/types.py
"""
Core type definitions for the evaluation framework.

This module contains the fundamental data structures used throughout
the hg_ds_evals library: ScoreLevel, Dimension, InputField, and OutputSchema.
"""


from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScoreLevel:
    """
    A single score level in an evaluation dimension's scale.
    
    ScoreLevels are immutable and define what each numeric score means
    for a particular dimension.
    
    Attributes:
        score: Integer score value (typically 0, 1, or 2).
        label: Short label for the score (e.g., "bad", "partial", "good").
        description: Detailed description of what this score means.
        
    Example:
        level = ScoreLevel(
            score=2, 
            label="good", 
            description="Fully meets expectations"
        )
    """
    score: int
    label: str
    description: str
    
    def __str__(self) -> str:
        return f"{self.score} ({self.label}): {self.description}"


@dataclass
class Dimension:
    """
    An evaluation dimension that can be composed into rubrics.
    
    Dimensions are the building blocks of rubrics. Each dimension
    represents one aspect of evaluation (e.g., "topic relevance",
    "information completeness").
    
    Attributes:
        id: Unique identifier for the dimension (snake_case recommended).
        name: Human-readable name for display.
        description: Question or statement the evaluator should assess.
        scale: Tuple of ScoreLevel objects defining the scoring scale.
        weight: Relative importance weight (default 1.0).
        
    Example:
        dim = Dimension(
            id="accuracy",
            name="Response Accuracy",
            description="Is the response factually correct?",
            scale=(
                ScoreLevel(0, "bad", "Contains errors"),
                ScoreLevel(1, "partial", "Mostly correct"),
                ScoreLevel(2, "good", "Fully accurate"),
            ),
            weight=1.5
        )
    """
    id: str
    name: str
    description: str
    scale: tuple[ScoreLevel, ...] = ()
    weight: float = 1.0

    def __post_init__(self):
        """Convert list to tuple for consistency."""
        if isinstance(self.scale, list):
            object.__setattr__(self, 'scale', tuple(self.scale))
    
    def with_weight(self, new_weight: float) -> 'Dimension':
        """
        Return a new Dimension with updated weight.
        
        Args:
            new_weight: The new weight value.
            
        Returns:
            New Dimension instance with the updated weight.
            
        Example:
            critical_dim = TOPIC_RELEVANCE.with_weight(2.0)
        """
        return Dimension(
            id=self.id,
            name=self.name,
            description=self.description,
            scale=self.scale,
            weight=new_weight
        )
    
    def with_custom_scale(self, scale: list[ScoreLevel]) -> "Dimension":
        """
        Return a new Dimension with custom scale descriptions.
        
        Args:
            scale: List of ScoreLevel objects for the new scale.
            
        Returns:
            New Dimension instance with the custom scale.
        """
        return Dimension(
            id=self.id,
            name=self.name,
            description=self.description,
            scale=tuple(scale),
            weight=self.weight,
        )
    
    def describe(self) -> str:
        """
        Return a formatted string describing this dimension.
        
        Returns:
            Multi-line string with dimension details and scale.
        """
        lines = [
            f"Dimension: {self.name}",
            f"  ID: {self.id}",
            f"  Weight: {self.weight}",
            f"  Description: {self.description}",
            f"  Scale:",
        ]
        for level in self.scale:
            lines.append(f"    {level.score} ({level.label}): {level.description}")
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"Dimension(id='{self.id}', name='{self.name}', weight={self.weight})"


# =============================================================================
# Input/Output Schema Types for Prompt Generation
# =============================================================================

@dataclass(frozen=True)
class InputField:
    """Definition of an input field expected in the user prompt.
    InputFields specify and describe the data that should be provided 
    to the LLM judge for evaluation. This information is used to 
    generate user prompts template dynamically.

    Attributes:
        name: Field name as it appears in the prompt (e.g., "user_query").
        description: Explanation of what this field contains.
        required: Whether this field must be present (default True).
        
    Example:
        field = InputField(
            name="user_query",
            description="The exact user message that triggered the evaluation",
            required=True
        )
    """
    name:str
    description:str
    required:bool = True

    def __str__(self) -> str:
        req = "required" if self.required else "optional"
        return f"{self.name} ({req}): {self.description}"


@dataclass
class OutputField:
    """
    Definition of an output field expected in the LLM response.
    
    Attributes:
        name: Field name in the JSON output.
        field_type: Expected type ("string", "integer", "boolean", "array", "object").
        description: Explanation of what this field should contain.
        enum: Optional list of allowed values for enum-type fields.
    """
    name: str
    field_type: str
    description: str
    enum: tuple[str, ...] = ()
    
    def __post_init__(self):
        if isinstance(self.enum, list):
            object.__setattr__(self, 'enum', tuple(self.enum))

@dataclass
class OutputSchema:
    """
    Schema definition for the expected LLM judge output.
    
    OutputSchema defines the structure of the JSON response expected from
    the LLM judge. It's used to generate the output format section of
    the system prompt and to validate responses.
    
    Attributes:
        fields: Tuple of OutputField objects defining the response structure.
        additional_instructions: Extra instructions for output formatting.
        
    Example:
        schema = OutputSchema(
            fields=(
                OutputField("fallback_justified", "boolean", "Was fallback appropriate?"),
                OutputField("root_cause", "string", "Primary reason", enum=("wrong_enum", "insufficient_depth")),
            )
        )
    """
    fields: tuple[OutputField, ...] = ()
    additional_instructions: str = ""

    def __post_init__(self):
        if isinstance(self.fields, list):
            object.__setattr__(self, 'fields', tuple(self.fields))
    
    def get_field(self, name: str) -> OutputField | None:
        """Get an output field by name."""
        for f in self.fields:
            if f.name == name:
                return f
        return None