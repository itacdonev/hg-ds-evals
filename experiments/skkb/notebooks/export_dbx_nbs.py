import json
import subprocess
from pathlib import Path

PROFILE = "adb-prod"
REMOTE_ROOT = "/Users/sg7cb@s-mxs.net/hg-ds-evals/experiments/skkb/notebooks"
LOCAL_ROOT = Path("/Users/SG7CB/Developer/hg-ds-evals/experiments/skkb/notebooks")
EXPORT_NON_NOTEBOOK_FILES = False

def run(*args):
    subprocess.run(args, check=True)

def get_json(*args):
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)

def export_workspace_path(remote_path: str, local_path: Path, export_format: str | None = None):
    if local_path.exists():
        local_path.unlink()

    command = [
        "databricks", "workspace", "export", remote_path,
        "--file", str(local_path),
        "--profile", PROFILE,
    ]
    if export_format is not None:
        command.extend(["--format", export_format])

    run(*command)

def export_tree(remote_path: str, local_dir: Path):
    local_dir.mkdir(parents=True, exist_ok=True)
    entries = get_json("databricks", "workspace", "list", remote_path, "--profile", PROFILE, "-o", "json")

    for entry in entries:
        path = entry["path"]
        object_type = entry["object_type"]
        name = Path(path).name

        if object_type == "DIRECTORY":
            export_tree(path, local_dir / name)
        elif object_type == "NOTEBOOK":
            export_workspace_path(path, local_dir / f"{name}.ipynb", "JUPYTER")
        elif object_type == "FILE" and EXPORT_NON_NOTEBOOK_FILES:
            export_workspace_path(path, local_dir / name)

export_tree(REMOTE_ROOT, LOCAL_ROOT)