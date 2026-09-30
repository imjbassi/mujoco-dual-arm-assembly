import json

import mujoco
import numpy as np
import pytest

from dual_arm_assembly import AssemblySimulation, Config
from dual_arm_assembly.cli import main
from dual_arm_assembly.reporting import save_run
from dual_arm_assembly.scene import build_scene


@pytest.mark.parametrize("seed", [0, 7, 19])
def test_randomized_assembly_completes(seed):
    sim = AssemblySimulation(Config(seed=seed, randomize=True))
    result = sim.run()
    assert result["success"]
    assert 36 < result["insertion_depth_mm"] < 44
    assert result["lateral_error_mm"] < 3
    assert result["stable_hold_s"] >= 0.5
    assert not np.any(sim.data.warning.number)
    assert {row["phase"] for row in sim.trace} == {
        "SETTLE",
        "TRANSFER",
        "ALIGN",
        "INSERT",
        "HOLD",
        "COMPLETE",
    }


def test_misalignment_produces_contact_and_force_abort():
    sim = AssemblySimulation(Config(offset_mm=12))
    result = sim.run()
    assert result["status"] == "force_limit"
    assert not result["success"]
    assert result["contact_steps"] > 0
    assert result["peak_contact_force_n"] > sim.config.force_limit
    assert result["insertion_depth_mm"] < 5
    frozen = sim.data.qpos.copy()
    sim.step()
    np.testing.assert_array_equal(sim.data.qpos, frozen)


def test_nominal_posture_clears_table_and_wrist_stays_above_peg():
    sim = AssemblySimulation()
    link_ids = [sim.model.body(f"{arm}_link{i}").id for arm in ("left", "right") for i in range(7)]
    while not sim.done:
        if sim.steps % 100 == 0:
            # Table top is 0.4 m; links have radii at most 0.041 m.
            assert np.min(sim.data.xpos[link_ids, 2]) > 0.445
            assert sim.data.body("right_link5").xpos[2] > sim.data.site("peg_tip").xpos[2] + 0.2
        sim.step()


def test_success_not_reported_for_off_center_insertion():
    sim = AssemblySimulation(Config(offset_mm=3.5))
    result = sim.run()
    assert result["status"] == "timeout"
    assert not result["success"]


def test_reset_is_reproducible_and_ik_does_not_mutate_live_state():
    sim = AssemblySimulation(Config(seed=42, randomize=True))
    initial = sim.data.qpos.copy()
    sim.right.solve(sim.data.qpos, sim.goal)
    np.testing.assert_array_equal(sim.data.qpos, initial)
    first = sim.run()
    sim.reset()
    np.testing.assert_array_equal(sim.data.qpos, initial)
    second = sim.run()
    assert first == second


def test_export_compiles_and_reports_are_serializable(tmp_path):
    xml = build_scene()
    model = mujoco.MjModel.from_xml_string(xml)
    assert model.nu == 14
    sim = AssemblySimulation()
    sim.run()
    save_run(sim, tmp_path)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["success"]
    assert "mujoco" in result["versions"]
    assert (tmp_path / "trace.csv").read_text().startswith("time,phase,")
    assert "<svg" in (tmp_path / "report.html").read_text()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"force_limit": 0},
        {"force_limit": float("nan")},
        {"offset_mm": float("inf")},
        {"offset_mm": 31},
        {"duration": 1},
        {"duration": float("nan")},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_progress_tracks_time_and_saturates():
    sim = AssemblySimulation()
    assert sim.progress == 0.0
    sim.run()
    assert 0 < sim.progress <= 1.0


def test_report_includes_axis_chart_and_threshold_guides(tmp_path):
    sim = AssemblySimulation()
    sim.run()
    save_run(sim, tmp_path)
    report = (tmp_path / "report.html").read_text()
    assert "Axis alignment" in report
    assert "3 mm success limit" in report
    assert "35 N abort" in report
    assert "BIMANUAL" in report


def test_cli_benchmark_writes_summary_statistics(tmp_path):
    assert main(["benchmark", "--trials", "2", "--output", str(tmp_path)]) == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["successes"] == 2
    assert summary["statuses"] == {"success": 2}
    stats = summary["stats"]["lateral_error_mm"]
    assert stats["min"] <= stats["median"] <= stats["max"]
    assert stats["max"] < 3


def test_cli_writes_run_and_export(tmp_path):
    assert main(["run", "--output", str(tmp_path / "trial")]) == 0
    assert main(["export", "--output", str(tmp_path / "scene.xml")]) == 0
    assert (tmp_path / "scene.xml").is_file()


def test_cli_failure_exit_code(tmp_path):
    assert main(["run", "--offset-mm", "12", "--output", str(tmp_path)]) == 1
