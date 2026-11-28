"""Deterministic workload-to-temperature integration checks for both solvers."""
from unittest.mock import patch

import numpy as np
import pytest

from src.grid_and_boundary import ChipParameters, create_chip_grid
from src.krylov_solvers import solve_cg, solve_pcg
from src.thermal_integration import ThermalSolverAdapter, workload_states_to_power_schedule
from src.sparse_assembly import assemble_system
from src.thermal_model import run_thermal_simulation


@pytest.mark.parametrize("method", ["cg", "pcg"])
def test_workload_states_flow_through_fvm_and_selected_krylov_solver(method, monkeypatch):
    rows, cols = 4, 5
    states = ["Low", "High", "Medium"]  # synthetic states; no dataset semantics assumed
    power_scale = 40000.0  # explicit illustrative W/m^2 scale, not a calibration
    heat_levels = {"Low": 0.25, "Medium": 0.5, "High": 1.0}
    schedule = workload_states_to_power_schedule(
        states,
        (rows, cols),
        power_scale,
        heat_levels=heat_levels,
    )
    np.testing.assert_allclose(
        [float(source.mean()) for source in schedule], [10000.0, 40000.0, 20000.0]
    )
    assert all(source.shape == (rows, cols) for source in schedule)

    grid = create_chip_grid(
        rows, cols, die_width=0.01, die_height=0.008, initial_temperature=300.0
    )
    adapter = ThermalSolverAdapter(
        method=method,
        warm_start=True,
        reuse_preconditioner=(method == "pcg"),
        tol=1e-10,
        max_iter=500,
    )

    def direct_solver_must_not_be_used(*args, **kwargs):
        raise AssertionError("Integrated run unexpectedly fell back to SciPy spsolve.")

    assembled_systems = []

    def capture_assembly(*args, **kwargs):
        matrix, rhs = assemble_system(*args, **kwargs)
        assembled_systems.append((matrix, rhs))
        return matrix, rhs

    monkeypatch.setattr("src.thermal_model.spsolve", direct_solver_must_not_be_used)
    monkeypatch.setattr("src.thermal_model.assemble_system", capture_assembly)
    solver_spy = solve_cg if method == "cg" else solve_pcg
    adapter_target = "src.thermal_integration.solve_cg" if method == "cg" else "src.thermal_integration.solve_pcg"
    with patch(adapter_target, wraps=solver_spy) as selected_solver:
        history = run_thermal_simulation(
            grid,
            ChipParameters(),
            steps=len(schedule),
            solver=adapter,
            power_schedule=schedule,
        )

    assert selected_solver.call_count == len(states)
    assert adapter.method == method
    assert history.shape == (len(states) + 1, rows, cols)
    assert np.all(np.isfinite(history))
    assert len(adapter.records) == len(states)
    assert len(assembled_systems) == len(states)
    assert adapter.summary["converged_steps"] == len(states)
    for record, (matrix, rhs) in zip(adapter.records, assembled_systems):
        direct_residual = np.linalg.norm(rhs - matrix @ record.result.solution)
        assert record.result.converged
        assert record.result.final_residual == pytest.approx(direct_residual, rel=1e-12, abs=1e-16)
        assert record.result.final_residual <= 1e-10 * max(np.linalg.norm(rhs), 1.0)

    temperature_increments = np.diff(history.mean(axis=(1, 2)))
    assert temperature_increments[1] > temperature_increments[0]
    assert temperature_increments[2] < temperature_increments[1]
