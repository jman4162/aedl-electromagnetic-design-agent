"""Tests for the sysml2kit bridge, against the published t3-001 thresholds."""

from types import SimpleNamespace

from aedl.interop import requirements_from_spec, requirements_from_specs

# Mirrors sysml2kit requirements_extract output for the rf-library t3-001
# model; bounds must match tasks/t3-001-satcom-terminal-28ghz/task.yaml.
T3001_SPECS = [
    {
        "id": "link-margin",
        "metric_key": "worst_case_link_margin_db",
        "minimum": 0.0,
        "maximum": None,
    },
    {
        "id": "sidelobes",
        "metric_key": "worst_case_pattern_sll_db",
        "minimum": None,
        "maximum": -16.0,
    },
    {"id": "prime-power", "metric_key": "prime_power_w", "minimum": None, "maximum": 450.0},
    {"id": "unit-cost", "metric_key": "unit_cost_usd", "minimum": None, "maximum": 45000.0},
    {"id": "grating", "metric_key": "grating_margin_lambda", "minimum": 0.0, "maximum": None},
]


def test_one_sided_specs_map_directly():
    reqs = requirements_from_specs(T3001_SPECS)
    by_id = {r.id: r for r in reqs}
    assert len(reqs) == 5
    assert by_id["link-margin"].min == 0.0
    assert by_id["sidelobes"].max == -16.0
    assert by_id["prime-power"].max == 450.0


def test_bounds_match_task_yaml():
    reqs = {r.id: r for r in requirements_from_specs(T3001_SPECS)}
    assert reqs["unit-cost"].limit == "<= 45000.0"
    assert reqs["grating"].limit == ">= 0.0"


def test_equality_splits_into_pair():
    pair = requirements_from_spec(
        {"id": "exact", "metric_key": "x", "minimum": 5.0, "maximum": 5.0}
    )
    assert [r.id for r in pair] == ["exact-lo", "exact-hi"]
    assert pair[0].min == 5.0
    assert pair[1].max == 5.0


def test_prose_only_spec_dropped():
    assert (
        requirements_from_spec(
            {"id": "prose", "metric_key": "n/a", "minimum": None, "maximum": None}
        )
        == ()
    )


def test_object_specs_accepted():
    spec = SimpleNamespace(id="r1", metric_key="m", minimum=1.0, maximum=None)
    (req,) = requirements_from_spec(spec)
    assert req.metric == "m"
    assert req.check(2.0)
    assert not req.check(0.5)
