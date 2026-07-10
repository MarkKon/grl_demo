from scripts.summarize_ablations import parse_k


def test_parse_k_accepts_integer_and_float_csv_values() -> None:
    assert parse_k("4") == 4
    assert parse_k("4.0") == 4
