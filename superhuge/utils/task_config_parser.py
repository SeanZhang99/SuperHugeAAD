import ast
from collections import Counter
from collections.abc import Generator
from copy import deepcopy
from itertools import product
import math
import operator
import os
from typing import Any, Sequence

import yaml


def _is_expand_marker(value: Any) -> bool:
    """True if ``value`` is an explicit ``{expand: <list | expr>}`` marker."""
    return (
        isinstance(value, dict)
        and set(value.keys()) == {"expand"}
        and isinstance(value["expand"], (list, str))
    )


_FLOAT_ARTIFACT_RUN = 8
"""A run of this many identical digits (0s or 9s) in a float repr marks a binary
float precision artifact, e.g. ``0.30000000000000004`` or ``0.09999999999999998``."""


def _clean_float(value: Any) -> Any:
    """Round away binary-float representation artifacts, leaving genuine values alone.

    A double computed by inexact arithmetic (e.g. ``0 + 3 * 0.1`` or ``0.4 - 0.1``)
    often reprs with a long run of identical digits (``0.30000000000000004``,
    ``0.09999999999999998``). Detection is purely based on that repr run
    (>= ``_FLOAT_ARTIFACT_RUN`` consecutive 0s or 9s in the fractional part);
    only then is the value rounded to 10 decimal places. Non-floats and floats
    without such a run are returned unchanged.
    """
    if not isinstance(value, float):
        return value
    s = repr(value)
    if "e" in s or "E" in s:
        return value
    frac = s.partition(".")[2]
    if len(frac) < _FLOAT_ARTIFACT_RUN:
        return value
    if any(ch * _FLOAT_ARTIFACT_RUN in frac for ch in ("0", "9")):
        return round(value, 10)
    return value


def _fmt_tag_value(value: Any) -> str:
    """Format an expanded value for experiment-name tags (no minus sign, compact floats)."""
    value = _clean_float(value)
    if isinstance(value, float):
        s = "%.10g" % value
    else:
        s = str(value)
    return s.replace("-", "_")


def _frange(a=0.0, b=None, step=None) -> list[float]:
    """range() with float support: values = start + k*step, k = 0..n-1 (excludes stop)."""
    if b is None:
        start, stop, step = 0.0, float(a), 1.0
    elif step is None:
        start, stop, step = float(a), float(b), 1.0
    else:
        start, stop, step = float(a), float(b), float(step)
    if step == 0:
        raise ValueError("range step must not be zero")
    n = math.floor((stop - start) / step + 1e-9) if step > 0 \
        else math.floor((start - stop) / (-step) + 1e-9)
    n = max(n, 0)
    return [_clean_float(start + k * step) for k in range(n)]


def _eval_const_expr(node: ast.AST, names: dict) -> Any:
    """Safely evaluate a restricted arithmetic expression (int/float literals, + - * / // % **,
    unary signs, and lookups of already-fixed expand values by leaf name)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        raise ValueError(
            "TASK_CONFIG_PARSER:EXPAND:UNKNOWN_REF: "
            f"expand expression references non-expandable key {node.id!r}"
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _eval_const_expr(node.operand, names)
        return _clean_float(-v) if isinstance(node.op, ast.USub) else _clean_float(v)
    if isinstance(node, ast.BinOp):
        left = _eval_const_expr(node.left, names)
        right = _eval_const_expr(node.right, names)
        ops = {
            ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod, ast.Pow: operator.pow,
        }
        op = ops.get(type(node.op))
        if op is None:
            raise ValueError(
                "TASK_CONFIG_PARSER:EXPAND:OP: "
                f"unsupported operator {type(node.op).__name__}"
            )
        return _clean_float(op(left, right))
    raise ValueError(
        "TASK_CONFIG_PARSER:EXPAND:EXPR: "
        f"unsupported expression element {type(node).__name__}"
    )


def _const_value(a: ast.AST):
    """Numeric value of an ast Constant or a signed Constant (handles ``-1``)."""
    if isinstance(a, ast.Constant) and isinstance(a.value, (int, float)) \
            and not isinstance(a.value, bool):
        return a.value
    if isinstance(a, ast.UnaryOp) and isinstance(a.op, (ast.USub, ast.UAdd)):
        v = _const_value(a.operand)
        return -v if isinstance(a.op, ast.USub) else v
    raise ValueError(
        "TASK_CONFIG_PARSER:EXPAND:RANGE_ARGS: "
        f"expected a numeric constant, got {type(a).__name__}"
    )


def _parse_marker_value(raw: Any, leaf_names: set[str]):
    """Classify a marker value.

    Returns ('axis', values) for lists / 'range(a,b,step)' strings, or
    ('derived', ast_node, deps) for expressions that reference other marked keys.
    Derived markers are zipped elementwise with the keys they reference.
    """
    if not isinstance(raw, str):
        return ("axis", [_clean_float(v) for v in raw])
    try:
        node = ast.parse(raw, mode="eval").body
    except SyntaxError as e:
        raise ValueError(
            "TASK_CONFIG_PARSER:EXPAND:VALUE: "
            f"invalid expand expression {raw!r}: {e}"
        ) from e
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id == "range":
        if node.keywords:
            raise ValueError(
                "TASK_CONFIG_PARSER:EXPAND:RANGE_ARGS: "
                f"range args must be constants, got: {raw!r}"
            )
        args = [_const_value(a) for a in node.args]
        try:
            if all(isinstance(v, int) for v in args):
                return ("axis", list(range(*args)))
            return ("axis", _frange(*args))
        except (TypeError, ValueError) as e:
            raise ValueError(
                "TASK_CONFIG_PARSER:EXPAND:RANGE_ARGS: "
                f"invalid range expression: {raw!r} ({e})"
            ) from e
    refs = sorted({n.id for n in ast.walk(node) if isinstance(n, ast.Name)})
    if refs:
        unknown = set(refs) - set(leaf_names)
        if unknown:
            raise ValueError(
                "TASK_CONFIG_PARSER:EXPAND:UNKNOWN_REF: "
                f"expand expression references non-expandable key(s): {sorted(unknown)}"
            )
        return ("derived", node, set(refs))
    raise ValueError(
        "TASK_CONFIG_PARSER:EXPAND:VALUE: "
        f"expand value must be a list, range(...) or an expression, got: {raw!r}"
    )


def _contains_expand_marker(value: Any) -> bool:
    """True if any nested ``{expand: ...}`` marker exists inside ``value``."""
    if _is_expand_marker(value):
        return True
    if isinstance(value, dict):
        return any(_contains_expand_marker(v) for v in value.values())
    if isinstance(value, list):
        return any(_contains_expand_marker(v) for v in value)
    return False


class ConfigExpander:
    """Expands explicitly-marked ``{expand: ...}`` leaves into one concrete config
    per combination. Markers may live anywhere in the dict/list structure (list
    positions are addressed by index). Marker values are:

    - a list, or a ``range(a[, b[, step]])`` string  -> a cartesian axis;
    - an expression string referencing other marked keys (e.g. ``"post_lag + 0.4"``
      with ``post_lag`` marked) -> a DERIVED value, zipped elementwise with the
      keys it references (no extra cartesian dimension).

    Configs without markers pass through unchanged.
    """

    _PAIR_KEYS = ("train_dataset_id", "test_dataset_id")

    def __init__(self, dataset_names: dict | None = None):
        self.dataset_names = dataset_names or {}

    def expand(
        self,
        data: dict,
        model: dict,
        cv_name: str,
        exclude: Sequence[Sequence[int]] = (),
        skip_equal: bool = True,
    ):
        """Yield ``(variant, name_part)`` pairs, where variant = {"data": ..., "model": ...}
        with every expand marker replaced by one concrete scalar value."""
        base = {"data": deepcopy(data), "model": deepcopy(model)}
        markers: list[tuple[tuple, Any]] = []
        self._collect_markers(base, (), markers)
        if not markers:
            yield base, cv_name
            return

        leaf_names = {path[-1] for path, _ in markers}
        leaf_counts = Counter(path[-1] for path, _ in markers)
        dup_leaves = {leaf for leaf, n in leaf_counts.items() if n > 1}

        axes, derived = [], []
        for path, raw in markers:
            kind, *rest = _parse_marker_value(raw, leaf_names)
            if kind == "axis":
                axes.append((path, rest[0]))
            else:
                derived.append((path, rest[0], rest[1]))

        # ambiguous derived references: a derived marker must not reference a leaf
        # name that appears on more than one marker (cannot tell which one it means)
        for _, _, deps in derived:
            bad = deps & dup_leaves
            if bad:
                raise ValueError(
                    "TASK_CONFIG_PARSER:EXPAND:AMBIGUOUS_REF: "
                    f"expand expression references duplicated key(s) {sorted(bad)}; "
                    "give the marked keys distinct names"
                )
        bad_pair = dup_leaves & set(self._PAIR_KEYS)
        if bad_pair:
            raise ValueError(
                "TASK_CONFIG_PARSER:EXPAND:DUPLICATE_PAIR: "
                f"pair key(s) {sorted(bad_pair)} marked in more than one place"
            )

        # topological order of derived markers (their deps must be fixed first)
        fixed = {p[-1] for p, _ in axes}
        ordered, remaining = [], list(derived)
        while remaining:
            progressed = False
            for d in list(remaining):
                if d[2] <= fixed:
                    ordered.append(d)
                    fixed.add(d[0][-1])
                    remaining.remove(d)
                    progressed = True
            if not progressed:
                raise ValueError(
                    "TASK_CONFIG_PARSER:EXPAND:CIRCULAR: "
                    "circular expand references among: "
                    + ", ".join(p[-1] for p, _, _ in remaining)
                )

        # unambiguous tag keys: duplicated leaves get a "_<occurrence>" suffix
        seen = Counter()
        tag_key_of = {}
        for path, _ in markers:
            leaf = path[-1]
            seen[leaf] += 1
            tag_key_of[path] = leaf if leaf not in dup_leaves \
                else f"{leaf}_{seen[leaf]}"

        paths = [p for p, _ in axes]
        value_lists = [v for _, v in axes]
        exclude_pairs = {tuple(p) for p in exclude}
        for combo in product(*value_lists):
            variant = deepcopy(base)
            values = {}
            path_values = {}
            for path, value in zip(paths, combo):
                self._set_at_path(variant, path, value)
                path_values[path] = value
                values[path[-1]] = value
            for path, expr, _ in ordered:
                value = _eval_const_expr(expr, values)
                self._set_at_path(variant, path, value)
                path_values[path] = value
                values[path[-1]] = value
            tags = [(tag_key_of[path], path_values[path]) for path, _ in markers]
            name_part = self._name_part(values, tags, cv_name, exclude_pairs, skip_equal)
            if name_part is None:
                continue
            yield variant, name_part

    def _collect_markers(self, node: Any, path: tuple, out: list) -> None:
        if _is_expand_marker(node):
            out.append((path, node["expand"]))  # keep raw: list or expression string
            return
        if isinstance(node, dict):
            for key, value in node.items():
                self._collect_markers(value, path + (key,), out)
        elif isinstance(node, list):
            for index, item in enumerate(node):
                self._collect_markers(item, path + (index,), out)

    @staticmethod
    def _set_at_path(node, path: tuple, value: Any) -> None:
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value

    def _name_part(
        self,
        values: dict,
        tags: list[tuple[str, Any]],
        cv_name: str,
        exclude_pairs: set[tuple],
        skip_equal: bool,
    ) -> str | None:
        pair_leaves = [k for k in values if k in self._PAIR_KEYS]
        is_pair = len(pair_leaves) == 2 and set(pair_leaves) == set(self._PAIR_KEYS)

        if is_pair:
            train, test = values["train_dataset_id"], values["test_dataset_id"]
            if not (isinstance(train, int) and isinstance(test, int)):
                raise ValueError(
                    "TASK_CONFIG_PARSER:EXPAND:PAIR_TYPE: "
                    f"train_dataset_id/test_dataset_id must be ints, got {train!r} / {test!r}"
                )
            if skip_equal and train == test:
                return None
            if (train, test) in exclude_pairs:
                return None
            base = f"from_{self._name_of(train)}_to_{self._name_of(test)}"
        else:
            base = cv_name

        generic = [t for t in tags if t[0] not in self._PAIR_KEYS]
        if generic:
            tags_str = [f"{k}{_fmt_tag_value(v)}" for k, v in generic]
            return f"{base}-{'_'.join(tags_str)}"
        return base

    def _name_of(self, dataset_id: int) -> str:
        return str(self.dataset_names.get(dataset_id, dataset_id))


class TaskConfigParser:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config: dict[str, dict[str, dict[str, Any]]] = self._load_config()
        self.dataset_names: dict = self.config.pop("dataset_names", None) or {}
        self.expander = ConfigExpander(self.dataset_names)
        self.tasks = self._get_enabled_tasks()
        self.cross_validations = self._get_enabled_cross_validations()
        self.last_temp_file = None

    def _load_config(self) -> dict:
        with open(self.config_path, "r") as file:
            return yaml.safe_load(file)

    def _get_enabled_tasks(self) -> dict[str, dict]:
        tasks: dict[str, dict] = {}
        for task_type, task_info in self.config.items():
            for task_name, task_details in task_info.get("tasks", {}).items():
                if task_details.get("enable", 0) == 1:
                    tasks.setdefault(task_type, {})[task_name] = task_details
        return tasks

    def _get_enabled_cross_validations(self) -> dict[str, dict]:
        cross_validations: dict[str, dict] = {}
        for task_type, task_info in self.config.items():
            for cv_name, cv_details in task_info.get("cross_validation", {}).items():
                if cv_details.get("enable", 0) == 1:
                    cross_validations.setdefault(task_type, {})[cv_name] = cv_details
        return cross_validations

    def _deep_merge_dicts(self, base: dict, *updates: dict) -> dict:
        merged = deepcopy(base)
        for update in updates:
            for key, value in update.items():
                if _is_expand_marker(value):
                    # expand markers are opaque: last-wins, never merged or concatenated
                    merged[key] = deepcopy(value)
                elif isinstance(value, dict) and key in merged:
                    merged[key] = self._deep_merge_dicts(merged[key], value)
                elif (
                    isinstance(value, Sequence)
                    and key in merged
                    and isinstance(merged[key], Sequence)
                    and not isinstance(merged[key], str)
                ):
                    merged[key] = list(merged[key]) + list(value)

                else:
                    merged[key] = value
        return merged

    def _dict_to_cli_args(self, prefix: str, data: dict) -> list[str]:
        if _contains_expand_marker(data):
            raise ValueError(
                "TASK_CONFIG_PARSER:EXPAND:UNEXPANDED_MARKER: "
                f"expand marker under '{prefix}' reached CLI emission: {data!r}"
            )
        args = []
        for key, value in data.items():
            if isinstance(value, dict):
                args.extend(self._dict_to_cli_args(f"{prefix}.{key}", value))
            else:
                args.append(f"{prefix}.{key}")
                args.append(str(value))
        return args

    def generate_configs(self) -> Generator[tuple[list[str], str], Any, None]:
        for task_type, task_details in self.tasks.items():
            general_data: dict = self.config[task_type]["general"]["data"]
            for task_name, task_detail in task_details.items():
                for cv_name, cv_details in self.cross_validations.get(
                    task_type, {}
                ).items():
                    n_folds = (
                        cv_details.get("data", {})
                        .get("init_args", {})
                        .get("n_folds", None)
                    )
                    if n_folds is None:
                        n_folds = (
                            task_detail.get("data", {})
                            .get("init_args", {})
                            .get("n_folds", None)
                        )
                    if n_folds is None:
                        n_folds = general_data.get("init_args", {}).get("n_folds", None)
                    assert n_folds is not None, "n_folds not specified."
                    task_data: dict = task_detail.get("data", {})
                    cv_data: dict = cv_details.get("data", {})
                    merged_data: dict = self._deep_merge_dicts(
                        general_data, task_data, cv_data
                    )

                    general_model: dict = self.config[task_type]["general"].get(
                        "model", {}
                    )
                    task_model: dict = task_detail.get("model", {})
                    cv_model: dict = cv_details.get("model", {})
                    merged_model: dict = self._deep_merge_dicts(
                        general_model, task_model, cv_model
                    )

                    for variant, name_part in self.expander.expand(
                        merged_data,
                        merged_model,
                        cv_name=cv_name,
                        exclude=cv_details.get("exclude", []),
                        skip_equal=cv_details.get("skip_equal", True),
                    ):
                        cli_args: list[str] = []
                        cli_args.extend(
                            self._dict_to_cli_args("--data", variant["data"])
                        )
                        cli_args.extend(
                            self._dict_to_cli_args("--model", variant["model"])
                        )

                        experiment_name = f"{task_type}-{task_name}-{name_part}"

                        yield cli_args, experiment_name


if __name__ == "__main__":
    config_path = r"C:/Users/Sean/Documents/Seafile/ZYMdeDocument/24-12-SuperHugeAAD/SuperHugeAAD/scripts/dnn/configs/task_config.yaml"
    parser = TaskConfigParser(config_path)

    for config_file in parser.generate_configs():
        print(f"Generated config file: {config_file}")
