"""Smoke tests for Step 4 mapping visualization composition."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from minexpy.mapping import (
    create_grid,
    interpolate,
    plot_map,
    prepare,
)


def _sample_raw_data() -> pd.DataFrame:
    """Create a small geochemical sample dataset for smoke tests."""
    return pd.DataFrame(
        {
            "x": [0.0, 30.0, 60.0, 0.0, 60.0, 30.0],
            "y": [0.0, 0.0, 0.0, 60.0, 60.0, 30.0],
            "Zn": [10.0, 14.0, 20.0, 16.0, 28.0, 18.0],
        }
    )


def test_plot_map_full_pipeline_smoke() -> None:
    """plot_map should run full pipeline from raw data and return fig/ax."""
    raw = _sample_raw_data()
    fig, ax = plot_map(
        data=raw,
        x_col="x",
        y_col="y",
        value_col="Zn",
        cell_size=10.0,
        title="Smoke Map",
    )

    assert fig is not None
    assert ax is not None
    plt.close(fig)


def test_plot_map_precomputed_smoke() -> None:
    """plot_map should accept precomputed interpolation output."""
    raw = _sample_raw_data()
    prepared, meta = prepare(raw, x_col="x", y_col="y", value_col="Zn")
    grid = create_grid(prepared, cell_size=10.0)
    result = interpolate(prepared, grid, method="idw")

    fig, ax = plot_map(
        prepared=prepared,
        prepare_metadata=meta,
        interpolation_result=result,
        title="Precomputed Smoke",
    )

    assert fig is not None
    assert ax is not None
    plt.close(fig)


def test_plot_map_mixed_mode_warns() -> None:
    """Mixed precomputed/upstream inputs should emit warning."""
    raw = _sample_raw_data()
    prepared, _ = prepare(raw, x_col="x", y_col="y", value_col="Zn")
    grid = create_grid(prepared, cell_size=10.0)
    result = interpolate(prepared, grid, method="idw")

    with pytest.warns(UserWarning, match="ignored"):
        fig, ax = plot_map(
            data=raw,
            x_col="x",
            y_col="y",
            value_col="Zn",
            cell_size=10.0,
            interpolation_result=result,
        )
    plt.close(fig)


def test_plot_map_elements_present_smoke() -> None:
    """Default map should include color-mapped artists and title."""
    raw = _sample_raw_data()
    fig, ax = plot_map(
        data=raw,
        x_col="x",
        y_col="y",
        value_col="Zn",
        cell_size=10.0,
        title_parts={"what": "Zn (ppm)", "where": "Area X", "when": "2026"},
    )

    assert ax.get_title() != ""
    assert len(ax.collections) > 0
    plt.close(fig)
