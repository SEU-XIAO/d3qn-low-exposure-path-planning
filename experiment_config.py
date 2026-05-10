from __future__ import annotations

import argparse
import json
import shutil
import sys
import tomllib
from pathlib import Path
from typing import Any


def add_config_args(parser: argparse.ArgumentParser, default_section: str) -> None:
    parser.add_argument("--config", type=str, default=None, help="TOML 实验配置文件路径")
    parser.add_argument(
        "--config-section",
        type=str,
        default=default_section,
        help=f"TOML 中读取的配置节，默认 `{default_section}`",
    )


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    default_section: str,
    argv: list[str] | None = None,
) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)

    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config", type=str, default=None)
    probe.add_argument("--config-section", type=str, default=default_section)
    known, _unknown = probe.parse_known_args(raw_argv)

    if not known.config:
        return parser.parse_args(raw_argv)

    config_tokens = _load_config_tokens(
        config_path=Path(known.config),
        section_name=known.config_section or default_section,
    )
    return parser.parse_args(config_tokens + raw_argv)


def dump_effective_config(
    output_dir: Path,
    args: argparse.Namespace,
    runtime: dict[str, Any] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "args": {k: _jsonable(v) for k, v in vars(args).items()},
        "runtime": {k: _jsonable(v) for k, v in (runtime or {}).items()},
    }
    config_path = output_dir / "effective_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)

    source_config = getattr(args, "config", None)
    if source_config:
        resolved = _resolve_config_path(Path(str(source_config)))
        if resolved.exists():
            shutil.copy2(resolved, output_dir / "source_config.toml")

    return config_path


def _load_config_tokens(config_path: Path, section_name: str) -> list[str]:
    config_path = _resolve_config_path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with config_path.open("rb") as f:
        data = tomllib.load(f)

    section = _resolve_section(data, section_name, seen=[])
    tokens: list[str] = []
    for key, value in section.items():
        if key in {"config", "config_section", "inherits"}:
            continue
        tokens.extend(_value_to_tokens(key, value))
    return tokens


def _resolve_config_path(config_path: Path) -> Path:
    if config_path.is_absolute() or config_path.exists():
        return config_path
    repo_root = Path(__file__).resolve().parent
    return repo_root / config_path


def _resolve_section(data: dict[str, Any], section_name: str, seen: list[str]) -> dict[str, Any]:
    if section_name in seen:
        chain = " -> ".join(seen + [section_name])
        raise ValueError(f"检测到配置继承循环: {chain}")

    section = _get_section(data, section_name)
    if not isinstance(section, dict):
        raise TypeError(f"配置节 `{section_name}` 必须是 table")

    merged: dict[str, Any] = {}
    inherits = section.get("inherits", [])
    if isinstance(inherits, str):
        inherits = [inherits]
    if inherits:
        if not isinstance(inherits, list) or not all(isinstance(x, str) for x in inherits):
            raise TypeError(f"`{section_name}.inherits` 必须是字符串或字符串数组")
        for parent_name in inherits:
            merged.update(_resolve_section(data, parent_name, seen=seen + [section_name]))

    for key, value in section.items():
        if key == "inherits":
            continue
        merged[key] = value
    return merged


def _get_section(data: dict[str, Any], section_name: str) -> Any:
    cur: Any = data
    for part in section_name.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"配置节不存在: {section_name}")
        cur = cur[part]
    return cur


def _value_to_tokens(key: str, value: Any) -> list[str]:
    if isinstance(value, dict):
        raise TypeError(f"配置项 `{key}` 不支持嵌套 table，请改成标量或数组")

    flag = f"--{key.replace('_', '-')}"
    if isinstance(value, bool):
        return [flag] if value else []
    if isinstance(value, list):
        tokens: list[str] = []
        for item in value:
            tokens.extend([flag, str(item)])
        return tokens
    return [flag, str(value)]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value
