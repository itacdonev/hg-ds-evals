"""
Prompt Builder for LLM-as-Judge evaluations.

This module provides the PromptBuilder class that renders system and user
prompts from Rubric configurations using Jinja2 templates.

Usage:
    from hg_ds_evals.prompts.builder import PromptBuilder
    from hg_ds_evals.rubrics.fallback import FALLBACK_RUBRIC
    
    builder = PromptBuilder(rubric=FALLBACK_RUBRIC)
    
    # Build system prompt (instructions for the judge)
    system_prompt = builder.build_system_prompt()
    
    # Build user prompt (the example to evaluate)
    user_prompt = builder.build_user_prompt({
        "user_query": "What is my card limit?",
        "bot_answer": "I don't have that information.",
        ...
    })
"""

from pathlib import Path
from typing import Optional, Any
from jinja2 import Environment, FileSystemLoader, BaseLoader, TemplateNotFound

from hg_ds_evals.rubrics.base import Rubric


# =============================================================================
# Default Templates (embedded)
# =============================================================================

DEFAULT_SYSTEM_TEMPLATE = """You are an expert evaluator for {{ rubric.metadata.name }}.

## Your Task

{{ rubric.metadata.description }}

{% if rubric.judge_instructions %}
## Additional Instructions

{{ rubric.judge_instructions }}
{% endif %}

## Inputs You Will Receive

You will receive the following fields:
{% for field in rubric.input_fields %}
* **{{ field.name }}**{% if not field.required %} (optional){% endif %}: {{ field.description }}
{% endfor %}

## Evaluation Rubrics

Rate each dimension on a scale of 0-2. For each dimension, provide a score and brief reasoning.

{% for dim in rubric.dimensions %}
### {{ loop.index }}. {{ dim.name }} ({{ dim.id }})

**Question**: {{ dim.description }}

{% for level in dim.scale %}
- **Score {{ level.score }} ({{ level.label }})**: {{ level.description }}
{% endfor %}

{% endfor %}

## Output Format

Return ONLY valid JSON with this exact structure. Do not include any text outside the JSON.

```json
{
  "rubric_scores": {
{% for dim in rubric.dimensions %}
    "{{ dim.id }}": {
      "score": 0,
      "reasoning": "Brief explanation (max 100 chars)"
    }{% if not loop.last %},{% endif %}

{% endfor %}
  }{% if rubric.output_schema %},
{% for field in rubric.output_schema.fields %}
  "{{ field.name }}": {% if field.field_type == "boolean" %}true{% elif field.field_type == "array" %}[]{% elif field.field_type == "object" %}{}{% elif field.field_type == "integer" %}0{% else %}"..."{% endif %}{% if not loop.last %},{% endif %}

{% endfor %}
{% endif %}
}
```

{% if rubric.output_schema and rubric.output_schema.additional_instructions %}
{{ rubric.output_schema.additional_instructions }}
{% endif %}

## Rules

1. Return ONLY valid JSON - no markdown, no explanations outside JSON.
2. Keep reasoning concise (under 100 characters per dimension).
3. Base scores strictly on the rubric definitions above.
4. If information is missing, note it in your reasoning.
5. Write all responses in English regardless of input language.
"""


DEFAULT_USER_TEMPLATE = """Evaluate this case:

{% for field in rubric.input_fields %}
**{{ field.name }}**: {{ example.get(field.name, 'N/A') }}
{% endfor %}

Analyze using the rubrics provided and return JSON only.
"""

# =============================================================================
# Prompt Builder Class
# =============================================================================

class PromptBuilder:
    """
        Builds system and user prompts from rubric configurations.
        
        PromptBuilder uses Jinja2 templates to render prompts. It supports:
        - Default embedded templates (no files required)
        - Custom file-based templates
        - Template inheritance and overrides
        
        Attributes:
            rubric: The Rubric instance to build prompts for.
            system_template_path: Optional path to custom system template.
            user_template_path: Optional path to custom user template.
            
        Example:
            # Using default templates
            builder = PromptBuilder(rubric=FALLBACK_RUBRIC)
            
            # Using custom templates
            builder = PromptBuilder(
                rubric=FALLBACK_RUBRIC,
                system_template_path=Path("templates/fallback_system.md.j2"),
                user_template_path=Path("templates/fallback_user.md.j2"),
            )
        """
    
    def __init__(
        self,
        rubric: Rubric,
        system_template_path: Optional[Path] = None,
        user_template_path: Optional[Path] = None,
        templates_dir: Optional[Path] = None,
    ):
        """
        Initialize the PromptBuilder.
        
        Args:
            rubric: The Rubric instance to build prompts for.
            system_template_path: Path to custom system prompt template (.j2 file).
            user_template_path: Path to custom user prompt template (.j2 file).
            templates_dir: Base directory for templates (for FileSystemLoader).
        """
        self.rubric = rubric
        self.system_template_path = system_template_path
        self.user_template_path = user_template_path
        
        # Jinja2 environment
        if templates_dir and templates_dir.exists():
            self._env = Environment(
                loader=FileSystemLoader(str(templates_dir)),
                trim_blocks=True,
                lstrip_blocks=True,
            )
        else:
            self._env = Environment(
                loader=BaseLoader(),
                trim_blocks=True,
                lstrip_blocks=True,
            )
        
        # Load templates
        self._system_template = self._load_template(
            system_template_path, 
            DEFAULT_SYSTEM_TEMPLATE,
            "system"
        )
        self._user_template = self._load_template(
            user_template_path,
            DEFAULT_USER_TEMPLATE,
            "user"
        )
    
    # Conventional template filenames looked up when a directory is provided
    _TEMPLATE_FILENAMES: dict[str, str] = {
        "system": "system.md.j2",
        "user": "user.md.j2",
    }

    def _load_template(
        self, 
        template_path: Optional[Path], 
        default_template: str,
        template_type: str
    ):
        """Load a template from file or use default.
        
        If ``template_path`` is a directory, the conventional filename
        (``system.md.j2`` / ``user.md.j2``) is appended automatically.
        """
        if template_path:
            template_path = Path(template_path)

            # If the path is a directory, resolve to the conventional file
            if template_path.is_dir():
                filename = self._TEMPLATE_FILENAMES.get(template_type)
                if filename is None:
                    raise ValueError(
                        f"Unknown template_type '{template_type}'. "
                        f"Expected one of {list(self._TEMPLATE_FILENAMES)}."
                    )
                template_path = template_path / filename

        if template_path and template_path.exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                template_str = f.read()
            return self._env.from_string(template_str)
        elif template_path:
            # Path provided but doesn't exist - try FileSystemLoader
            try:
                return self._env.get_template(str(template_path))
            except TemplateNotFound:
                raise FileNotFoundError(
                    f"Template not found: {template_path}. "
                    f"Provide a valid path or use default templates."
                )
        else:
            # Use default embedded template
            return self._env.from_string(default_template)
    
    def build_system_prompt(self) -> str:
        """
        Build the system prompt for the LLM judge.
        
        The system prompt contains:
        - Task description from rubric metadata
        - Input field descriptions
        - Evaluation rubrics (dimensions with scales)
        - Output format specification
        - Additional judge instructions
        
        Returns:
            Rendered system prompt string.
            
        Example:
            system_prompt = builder.build_system_prompt()
            # Use with LLM: messages=[{"role": "system", "content": system_prompt}]
        """
        context = self._build_context()
        return self._system_template.render(**context)
    
    def build_user_prompt(self, example: dict[str, Any]) -> str:
        """
        Build the user prompt containing the example to evaluate.
        
        Args:
            example: Dictionary with field values to evaluate.
                     Keys should match the rubric's input_field names.
                     
        Returns:
            Rendered user prompt string.
            
        Example:
            user_prompt = builder.build_user_prompt({
                "user_query": "What is my card limit?",
                "bot_answer": "I don't have that information.",
                "conversation_history": "...",
            })
        """
        context = self._build_context()
        context["example"] = example
        return self._user_template.render(**context)
    
    def _build_context(self) -> dict[str, Any]:
        """Build the template context from the rubric."""
        return {
            "rubric": self.rubric,
        }
    
    def validate_example(self, example: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Check that an example contains all required input fields.
        
        Args:
            example: Dictionary with field values to validate.
            
        Returns:
            Tuple of (is_valid, list of error messages).
            
        Example:
            is_valid, errors = builder.validate_example(my_example)
            if not is_valid:
                print("Missing fields:", errors)
        """
        errors = []
        
        for field in self.rubric.input_fields:
            if field.required and field.name not in example:
                errors.append(f"Missing required field: {field.name}")
            elif field.required and example.get(field.name) in (None, "", "N/A"):
                errors.append(f"Required field '{field.name}' is empty")
        
        return len(errors) == 0, errors
    
    def get_template_info(self) -> dict[str, Any]:
        """
        Get information about the templates being used.
        
        Returns:
            Dictionary with template configuration details.
        """
        return {
            "rubric_id": self.rubric.metadata.id,
            "rubric_version": self.rubric.metadata.version,
            "system_template": str(self.system_template_path) if self.system_template_path else "default",
            "user_template": str(self.user_template_path) if self.user_template_path else "default",
            "input_fields": self.rubric.input_field_names,
            "dimensions": self.rubric.dimension_ids,
        }
    
    def __repr__(self) -> str:
        return (
            f"PromptBuilder(rubric='{self.rubric.metadata.id}', "
            f"dimensions={len(self.rubric.dimensions)}, "
            f"input_fields={len(self.rubric.input_fields)})"
        )
