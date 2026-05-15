# HEY GEORGE - EVALS
# General functions for prompt operations.

# prompts/common.py
#=============================================================

try:
    # Databricks environment
    displayHTML
except NameError:
    # Fallback for non-Databricks environments
    from IPython.display import HTML, display
    def displayHTML(html):
        display(HTML(html))


def display_prompt(
    prompt: str, 
    title: str = "Prompt", 
    font_size: int = 12,
    max_height: int = 600,
    background_color: str = "#d7e6ee"
) -> None:
    """
    Display a formatted prompt in Databricks notebook using HTML.
    
    Args:
        prompt: The prompt text to display.
        title: Title to show above the prompt (default: "Prompt").
        max_height: Maximum height in pixels before scrolling (default: 600).
        background_color: Background color for the display box (default: "#f5f5f5").
        
    Example:
        from hg_ds_evals.prompts.builder import PromptBuilder
        from hg_ds_evals.prompts.common import display_prompt
        
        builder = PromptBuilder(rubric=FALLBACK_RUBRIC)
        system_prompt = builder.build_system_prompt()
        
        display_prompt(system_prompt, title="System Prompt")
        
        user_prompt = builder.build_user_prompt(example_data)
        display_prompt(user_prompt, title="User Prompt", max_height=400)
    """
    displayHTML(f"""
    <div style="background-color: {background_color}; padding: 20px; border-radius: 5px; border: 1px solid #ddd; max-height: {max_height}px; overflow-y: auto;">
        <h3 style="margin-top: 0; color: #333;">{title}</h3>
        <pre style="white-space: pre-wrap; word-wrap: break-word; font-family: 'Courier New', monospace; font-size: {font_size}px; margin: 0;">
{prompt}
        </pre>
    </div>
    """
)

def read_md_file(file_path: str) -> str:
    """
    Reads and returns the contents of a markdown file.
    This was the old version of the system prompt reading capability.
    Left here for reference and potential use in fast prototyping of new prompts.
    """
    with open(file_path, "r") as f:
        return f.read()