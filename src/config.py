import os
import re
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _substitute_env_vars(value):
    if isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env_vars(v) for v in value]
    if isinstance(value, str):
        match = _ENV_VAR_PATTERN.fullmatch(value)
        if match:
            env_var_name = match.group(1)
            return os.environ.get(env_var_name)
    return value


@lru_cache(maxsize=1)
def load_config() -> dict:
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    with open(CONFIG_PATH, "r") as f:
        raw = yaml.safe_load(f)

    return _substitute_env_vars(raw)


def get_artifact_path(filename_key: str) -> Path:
    """Resolve one of the artifact filenames in config to a full path.
    e.g. get_artifact_path("model_file") -> PROJECT_ROOT/artifacts/model.pkl
    """
    config = load_config()
    artifacts_cfg = config["artifacts"]
    artifacts_dir = PROJECT_ROOT / artifacts_cfg["dir"]
    return artifacts_dir / artifacts_cfg[filename_key]
