"""
YAML Configuration Utilities
============================
Purpose:
    Provides lightweight YAML configuration loading and placeholder resolution.

Functions:
    - resolve_placeholders(cfg): Replaces `${section.key}` syntax with actual values from cfg.
    - load_yaml(path): Loads a YAML file into a Python dict.

Use Case:
    Used across pipeline scripts (e.g., `build_word_trends.py`) to interpret  
    path variables and dynamic settings from configuration files.
"""

from __future__ import annotations
import re, copy

_VAR = re.compile(r"\$\{([^}]+)\}")

def resolve_placeholders(cfg: dict) -> dict:
    """Resolve ${section.key} placeholders inside a nested dict."""
    def get_key(d, dotted, default=None):
        cur = d
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def subst(s: str) -> str:
        def repl(m):
            key = m.group(1)
            val = get_key(cfg, key, m.group(0))
            return str(val)
        return _VAR.sub(repl, s)

    def walk(o):
        if isinstance(o, dict):
            return {k: walk(v) for k, v in o.items()}
        if isinstance(o, list):
            return [walk(v) for v in o]
        if isinstance(o, str):
            return subst(o)
        return o

    # run on a deep copy so we can resolve against original names
    return walk(copy.deepcopy(cfg))

import yaml

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
