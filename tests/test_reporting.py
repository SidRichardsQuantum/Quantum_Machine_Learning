from qml.reporting import format_table, print_section


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
