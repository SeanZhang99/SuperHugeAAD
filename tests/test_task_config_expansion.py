"""Tests for the {expand: ...} config expansion feature in TaskConfigParser.

The parser module is loaded by file path so the tests run without importing the
full ``superhuge`` package (which pulls in torch).
"""
import importlib.util
import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_PARSER_PATH = _REPO / "superhuge" / "utils" / "task_config_parser.py"


def _load_parser():
    spec = importlib.util.spec_from_file_location(
        "task_config_parser_under_test", _PARSER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_tcp = _load_parser()
TaskConfigParser = _tcp.TaskConfigParser

DATASETS = [2, 4, 5, 12, 13, 14, 15]
NAMES = {2: "dtu", 4: "kul", 5: "avgc", 12: "ustc", 13: "njus", 14: "hd", 15: "dl"}

EXPANDED_CONFIG = """\
dataset_names: {2: dtu, 4: kul, 5: avgc, 12: ustc, 13: njus, 14: hd, 15: dl}
multi_data_sr:
  type: regression
  general:
    data:
      init_args:
        n_folds: 5
    model:
      init_args:
        num_audio_features:
          env: 1
        model_common_args:
          num_channels: 64
  tasks:
    pcc_diff:
      enable: 1
      model:
        init_args:
          loss:
            class_path: superhuge.model.loss.ContrastivePearsonLoss
  cross_validation:
    cross_dataset:
      enable: 1
      data:
        init_args:
          meta_group_func:
            class_path: utils.cross_validation.CrossDatasetValidation_
            init_args:
              train_dataset_id: {expand: [2, 4, 5, 12, 13, 14, 15]}
              test_dataset_id:  {expand: [2, 4, 5, 12, 13, 14, 15]}
"""


def write_config(tmp_path, text):
    path = tmp_path / "task_config.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def collect(tmp_path, text):
    parser = TaskConfigParser(write_config(tmp_path, text))
    return list(parser.generate_configs())


# ---------------------------------------------------------------- pair expansion


def test_pair_expansion_produces_42_ordered_configs(tmp_path):
    rows = collect(tmp_path, EXPANDED_CONFIG)
    assert len(rows) == 42
    expected_pairs = [
        (train, test)
        for train in DATASETS
        for test in DATASETS
        if train != test
    ]
    got_pairs = []
    for args, name in rows:
        assert name.startswith("multi_data_sr-pcc_diff-from_")
        assert name.endswith("_to_" + NAMES[int(args[args.index(
            "--data.init_args.meta_group_func.init_args.test_dataset_id") + 1])])
        train = int(args[args.index(
            "--data.init_args.meta_group_func.init_args.train_dataset_id") + 1])
        test = int(args[args.index(
            "--data.init_args.meta_group_func.init_args.test_dataset_id") + 1])
        got_pairs.append((train, test))
    assert got_pairs == expected_pairs  # order preserved: train-major


def test_pair_names_use_dataset_map(tmp_path):
    rows = collect(tmp_path, EXPANDED_CONFIG)
    names = [name for _, name in rows]
    assert "multi_data_sr-pcc_diff-from_hd_to_dtu" in names
    assert "multi_data_sr-pcc_diff-from_dl_to_kul" in names
    assert len(set(names)) == 42  # no duplicates


def test_numeric_fallback_without_dataset_names(tmp_path):
    text = EXPANDED_CONFIG.replace(
        "dataset_names: {2: dtu, 4: kul, 5: avgc, 12: ustc, 13: njus, 14: hd, 15: dl}\n",
        "",
    )
    rows = collect(tmp_path, text)
    names = [name for _, name in rows]
    assert "multi_data_sr-pcc_diff-from_2_to_4" in names
    assert "multi_data_sr-pcc_diff-from_15_to_14" in names


def test_skip_equal_false_keeps_49(tmp_path):
    text = EXPANDED_CONFIG.replace(
        "    cross_dataset:\n", "    cross_dataset:\n      skip_equal: false\n"
    )
    rows = collect(tmp_path, text)
    assert len(rows) == 49
    assert "multi_data_sr-pcc_diff-from_hd_to_hd" in [n for _, n in rows]


def test_exclude_drops_specific_pairs(tmp_path):
    text = EXPANDED_CONFIG.replace(
        "    cross_dataset:\n", "    cross_dataset:\n      exclude: [[2, 12], [14, 4]]\n"
    )
    rows = collect(tmp_path, text)
    assert len(rows) == 40
    names = [n for _, n in rows]
    assert "multi_data_sr-pcc_diff-from_dtu_to_ustc" not in names
    assert "multi_data_sr-pcc_diff-from_hd_to_kul" not in names
    assert "multi_data_sr-pcc_diff-from_dtu_to_kul" in names


# ------------------------------------------------------------ range expressions


def test_range_expression_expansion(tmp_path):
    text = """\
multi_data_sr:
  type: regression
  general:
    data:
      init_args:
        n_folds: 5
    model: {}
  tasks:
    t:
      enable: 1
  cross_validation:
    cv:
      enable: 1
      data:
        init_args:
          meta_group_func:
            class_path: utils.cross_validation.CrossDatasetValidation_
            init_args:
              train_dataset_id: {expand: "range(2, 5)"}
              test_dataset_id:  {expand: "range(4, 7)"}
"""
    rows = collect(tmp_path, text)
    # 3x3 combos minus (4, 4)
    assert len(rows) == 8
    trains, tests = set(), set()
    for _, name in rows:
        m = re.fullmatch(r"multi_data_sr-t-from_(\d+)_to_(\d+)", name)
        assert m
        trains.add(int(m.group(1)))
        tests.add(int(m.group(2)))
        assert m.group(1) != m.group(2)
    assert trains == {2, 3, 4}
    assert tests == {4, 5, 6}


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os')",
        "range(2.0, 5)",
        "range(1, 2, 3, 4)",
        "2:5",
        "range(1, 2, step=3)",
    ],
)
def test_range_rejects_unsafe_or_invalid_expressions(tmp_path, expr):
    text = """\
multi_data_sr:
  type: regression
  general:
    data:
      init_args:
        n_folds: 5
    model: {}
  tasks:
    t:
      enable: 1
  cross_validation:
    cv:
      enable: 1
      data:
        init_args:
          meta_group_func:
            class_path: utils.cross_validation.CrossDatasetValidation_
            init_args:
              train_dataset_id: {expand: [2, 4]}
              test_dataset_id:  {expand: "%s"}
""" % expr
    with pytest.raises(ValueError):
        collect(tmp_path, text)


# ------------------------------------------------------ generic task expansion


def test_generic_expansion_in_task(tmp_path):
    text = """\
multi_data_sr:
  type: regression
  general:
    data:
      init_args:
        n_folds: 5
    model: {}
  tasks:
    t:
      enable: 1
      data:
        init_args:
          window_length: {expand: [10, 30]}
  cross_validation:
    cv:
      enable: 1
"""
    rows = collect(tmp_path, text)
    assert len(rows) == 2
    names = [n for _, n in rows]
    assert "multi_data_sr-t-cv-window_length10" in names
    assert "multi_data_sr-t-cv-window_length30" in names
    for args, _ in rows:
        assert "--data.init_args.window_length" in args


def test_pair_plus_generic_combines_names(tmp_path):
    text = """\
multi_data_sr:
  type: regression
  general:
    data:
      init_args:
        n_folds: 5
    model: {}
  tasks:
    t:
      enable: 1
      data:
        init_args:
          window_length: {expand: [10, 30]}
  cross_validation:
    cv:
      enable: 1
      data:
        init_args:
          meta_group_func:
            class_path: utils.cross_validation.CrossDatasetValidation_
            init_args:
              train_dataset_id: {expand: [2, 4]}
              test_dataset_id:  {expand: [4, 5]}
"""
    rows = collect(tmp_path, text)
    # pairs (2,4) (2,5) (4,5) [skip (4,4)] x 2 windows
    assert len(rows) == 6
    names = [n for _, n in rows]
    assert len(set(names)) == 6
    assert "multi_data_sr-t-from_2_to_4-window_length10" in names
    assert "multi_data_sr-t-from_4_to_5-window_length30" in names


# --------------------------------------------------- zipped (derived) expansion


def test_zip_derived_expression(tmp_path):
    # post_lag = pre_lag + 0.4: zipped one-to-one, NOT cartesian
    text = """\
multi_data_sr:
  type: regression
  general:
    data:
      init_args:
        n_folds: 5
    model: {}
  tasks:
    t:
      enable: 1
  cross_validation:
    lag_search:
      enable: 1
      data:
        init_args:
          pre_lag: {expand: "range(-1, 1, 0.1)"}
          post_lag: {expand: "pre_lag + 0.4"}
"""
    rows = collect(tmp_path, text)
    assert len(rows) == 20  # zipped, not 20*20
    names = [n for _, n in rows]
    assert len(set(names)) == 20
    pairs = []
    for args, name in rows:
        pre = float(args[args.index("--data.init_args.pre_lag") + 1])
        post = float(args[args.index("--data.init_args.post_lag") + 1])
        pairs.append((pre, post))
        assert abs(post - (pre + 0.4)) < 1e-6
        assert "pre_lag" in name and "post_lag" in name
    pres = sorted(p for p, _ in pairs)
    assert len(pres) == 20 and abs(pres[0] - (-1.0)) < 1e-9
    assert all(p1 != p2 for p1, p2 in zip(pairs, pairs[1:]))  # distinct pairs


def test_zip_derived_chain(tmp_path):
    text = """\
multi_data_sr:
  type: regression
  general:
    data:
      init_args:
        n_folds: 5
    model: {}
  tasks:
    t:
      enable: 1
  cross_validation:
    lag_search:
      enable: 1
      data:
        init_args:
          pre_lag: {expand: [0.0, 0.2, 0.4]}
          post_lag: {expand: "pre_lag + 0.4"}
          center: {expand: "(pre_lag + post_lag) / 2"}
"""
    rows = collect(tmp_path, text)
    assert len(rows) == 3  # chain adds no dimensions
    for args, name in rows:
        pre = float(args[args.index("--data.init_args.pre_lag") + 1])
        post = float(args[args.index("--data.init_args.post_lag") + 1])
        cen = float(args[args.index("--data.init_args.center") + 1])
        assert abs(cen - (pre + post) / 2) < 1e-9


def test_zip_unknown_ref_raises(tmp_path):
    text = """\
multi_data_sr:
  type: regression
  general:
    data:
      init_args:
        n_folds: 5
    model: {}
  tasks:
    t:
      enable: 1
  cross_validation:
    cv:
      enable: 1
      data:
        init_args:
          pre_lag: {expand: [0.0, 0.2]}
          post_lag: {expand: "foo + 0.4"}
"""
    with pytest.raises(ValueError, match="UNKNOWN_REF"):
        collect(tmp_path, text)


def test_zip_circular_ref_raises(tmp_path):
    text = """\
multi_data_sr:
  type: regression
  general:
    data:
      init_args:
        n_folds: 5
    model: {}
  tasks:
    t:
      enable: 1
  cross_validation:
    cv:
      enable: 1
      data:
        init_args:
          a: {expand: "b + 1"}
          b: {expand: "a + 1"}
"""
    with pytest.raises(ValueError, match="CIRCULAR"):
        collect(tmp_path, text)


def test_duplicate_leaf_markers_unique_names(tmp_path):
    # two metadata filters, each with an expanded attribute_value (same leaf name)
    text = """\
multi_data_sr:
  type: regression
  general:
    data:
      init_args:
        n_folds: 5
    model: {}
  tasks:
    t:
      enable: 1
  cross_validation:
    cv:
      enable: 1
      data:
        init_args:
          add_meta_filter_func:
            - init_args:
                attribute_name: dataset_id
                attribute_value: {expand: [2, 4]}
            - init_args:
                attribute_name: subject_id
                attribute_value: {expand: [10, 20]}
"""
    rows = collect(tmp_path, text)
    assert len(rows) == 4  # 2x2 cross product
    names = [n for _, n in rows]
    assert len(set(names)) == 4  # no name collision
    for args, _ in rows:
        filters = args[args.index("--data.init_args.add_meta_filter_func") + 1]
        ds = int(filters.split("'attribute_name': 'dataset_id'")[1]
                 .split("attribute_value': ")[1].split("}")[0])
        sj = int(filters.split("'attribute_name': 'subject_id'")[1]
                 .split("attribute_value': ")[1].split("}")[0])
        assert ds in (2, 4) and sj in (10, 20)


def test_ambiguous_derived_ref_raises(tmp_path):
    # a derived marker referencing a leaf name that appears on two markers
    text = """\
multi_data_sr:
  type: regression
  general:
    data:
      init_args:
        n_folds: 5
    model: {}
  tasks:
    t:
      enable: 1
  cross_validation:
    cv:
      enable: 1
      data:
        init_args:
          add_meta_filter_func:
            - init_args:
                attribute_value: {expand: [1, 2]}
            - init_args:
                attribute_value: {expand: [3, 4]}
          scaled: {expand: "attribute_value * 10"}
"""
    with pytest.raises(ValueError, match="AMBIGUOUS_REF"):
        collect(tmp_path, text)


# -------------------------------------------------- identity & guard behavior


def test_no_marker_identity(tmp_path):
    text = """\
multi_data_sr:
  type: regression
  general:
    data:
      init_args:
        n_folds: 5
    model:
      init_args:
        num_audio_features:
          env: 1
  tasks:
    t:
      enable: 1
  cross_validation:
    from_2_to_4:
      enable: 1
      data:
        init_args:
          meta_group_func:
            class_path: utils.cross_validation.CrossDatasetValidation_
            init_args:
              train_dataset_id: 2
              test_dataset_id: 4
"""
    rows = collect(tmp_path, text)
    assert len(rows) == 1
    args, name = rows[0]
    assert name == "multi_data_sr-t-from_2_to_4"
    assert "--data.init_args.n_folds" in args and "5" in args
    assert "--data.init_args.meta_group_func.init_args.train_dataset_id" in args


def test_marker_inside_list_expands(tmp_path):
    # marker nested in a list element (metadata filter pattern): expands per value
    text = """\
multi_data_sr:
  type: regression
  general:
    data:
      init_args:
        n_folds: 5
    model: {}
  tasks:
    t:
      enable: 1
  cross_validation:
    cv:
      enable: 1
      data:
        init_args:
          add_meta_filter_func:
            - class_path: a
              init_args:
                attribute_name: dataset_id
                attribute_value: 4
            - class_path: b
              init_args:
                attribute_name: subject_id
                attribute_value: {expand: "range(1, 4)"}
"""
    rows = collect(tmp_path, text)
    assert len(rows) == 3
    names = [n for _, n in rows]
    assert "multi_data_sr-t-cv-attribute_value1" in names
    assert "multi_data_sr-t-cv-attribute_value3" in names
    for args, name in rows:
        filters = args[args.index("--data.init_args.add_meta_filter_func") + 1]
        subject = int(re.search(r"(\d+)$", name).group(1))
        assert f"'attribute_name': 'subject_id', 'attribute_value': {subject}" in filters
        assert "'attribute_name': 'dataset_id', 'attribute_value': 4" in filters


def test_cli_guard_rejects_surviving_marker(tmp_path):
    # direct guard check: a marker that somehow survives must raise, never emit
    parser = TaskConfigParser(write_config(tmp_path, "multi_data_sr:\n  type: x\n"))
    with pytest.raises(ValueError, match="UNEXPANDED_MARKER"):
        parser._dict_to_cli_args("--data", {"init_args": {"filters": [{"expand": [1, 2]}]}})


def test_loso_config_unchanged():
    loso_config = _REPO / "train" / "configs" / "task_config.yaml"
    if not loso_config.is_file():
        pytest.skip("loso config not present in this checkout")
    rows = list(TaskConfigParser(str(loso_config)).generate_configs())
    assert len(rows) == 2
    for args, name in rows:
        assert "expand" not in name
        assert not any("expand" in a for a in args)


# ------------------------------------------- float artifact cleanup (8-zero/9 run)


def test_clean_float_rounds_artifacts_only():
    clean = _tcp._clean_float
    # binary-float artifacts -> rounded to 10 dp
    assert clean(0.30000000000000004) == 0.3
    assert clean(0.09999999999999998) == 0.1
    assert clean(0.6000000000000001) == 0.6
    assert clean(-0.29999999999999993) == -0.3
    # genuine values / non-float pass through untouched
    assert clean(0.1) == 0.1
    assert clean(0.4) == 0.4
    assert clean(0.12345678901234566) == 0.12345678901234566
    assert clean(1e-08) == 1e-08  # scientific repr, never flagged
    assert clean(5.0) == 5.0
    assert clean(123.45678901234567) == 123.45678901234567
    assert clean(2) == 2
    assert clean("0.30000000000000004") == "0.30000000000000004"


def test_float_range_emits_clean_values():
    assert _tcp._frange(0, 0.5, 0.1) == [0.0, 0.1, 0.2, 0.3, 0.4]
    fr = _tcp._frange(-1, 1, 0.1)
    assert fr == pytest.approx([-1.0 + 0.1 * k for k in range(20)])
    assert not any("0000000000" in repr(v) or "9999999999" in repr(v) for v in fr)


def test_float_artifacts_cleaned_in_names_and_args(tmp_path):
    # user scenario: range axis + derived 0.4 - pre_lag must not leak
    # 0.30000000000000004-style artifacts into experiment names or CLI args
    text = """\
multi_data_sr:
  type: regression
  general:
    data:
      init_args:
        n_folds: 5
    model: {}
  tasks:
    t:
      enable: 1
  cross_validation:
    lag_search:
      enable: 1
      data:
        init_args:
          pre_lag: {expand: "range(0, 0.5, 0.1)"}
          post_lag: {expand: "0.4 - pre_lag"}
"""
    rows = collect(tmp_path, text)
    assert len(rows) == 5
    for args, name in rows:
        assert "0000000000000" not in name and "9999999999999" not in name
        pre = args[args.index("--data.init_args.pre_lag") + 1]
        post = args[args.index("--data.init_args.post_lag") + 1]
        assert re.fullmatch(r"-?\d+(\.\d+)?", pre), pre
        assert re.fullmatch(r"-?\d+(\.\d+)?", post), post
        assert abs(float(post) - (0.4 - float(pre))) < 1e-9
    names = [n for _, n in rows]
    assert len(set(names)) == 5
    assert any(n.endswith("pre_lag0.3_post_lag0.1") for n in names)
    assert any(n.endswith("pre_lag0.1_post_lag0.3") for n in names)
