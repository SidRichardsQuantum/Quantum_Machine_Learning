from qml.reporting import format_table, model_selection_table, print_section


def test_format_table_with_tuple_rows():
    table = format_table(
        [("Accuracy", 0.987654321), ("Passed", True)],
        title="Results",
    )

    assert "Results" in table
    assert "| Metric   | Value    |" in table
    assert "| Accuracy | 0.987654 |" in table
    assert "| Passed   | True     |" in table


def test_format_table_with_dict_rows_and_columns():
    table = format_table(
        [
            {"model": "quantum", "mae": 0.123456},
            {"model": "ridge", "mae": 0.05},
        ],
        columns=["model", "mae"],
        float_digits=4,
    )

    assert "| model   | mae    |" in table
    assert "| quantum | 0.1235 |" in table
    assert "| ridge   | 0.05   |" in table


def test_print_section_outputs_table(capsys):
    print_section("Dataset", [("Samples", 12), ("Features", ["x0", "x1"])])

    captured = capsys.readouterr()
    assert "Dataset" in captured.out
    assert "| Samples  | 12       |" in captured.out
    assert "| Features | [x0, x1] |" in captured.out


def test_model_selection_table_formats_candidate_summary():
    table = model_selection_table(
        {
            "scoring": "accuracy",
            "best_name": "reservoir",
            "candidates": [
                {
                    "name": "linear",
                    "cv_result": {
                        "summary": {
                            "test_score": {
                                "mean": 0.75,
                                "ci95_low": 0.7,
                                "ci95_high": 0.8,
                            },
                            "fit_seconds": {"mean": 0.01},
                        }
                    },
                },
                {
                    "name": "reservoir",
                    "cv_result": {
                        "summary": {
                            "test_score": {
                                "mean": 0.8,
                                "ci95_low": 0.76,
                                "ci95_high": 0.84,
                            },
                            "fit_seconds": {"mean": 0.2},
                        }
                    },
                },
            ],
        },
        title="Model selection",
    )

    assert "Model selection" in table
    assert "reservoir" in table
    assert "True" in table
