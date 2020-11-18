import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
from cjwmodule.i18n import I18nMessage
from cjwmodule.testing.i18n import cjwmodule_i18n_message, i18n_message
from cjwmodule.util.colnames import DefaultSettings, Settings

from labelbycondition import render


def EQ(column: str, value: int) -> Dict[str, Any]:
    return dict(operation="number_is", column=column, value=value)


def GT(column: str, value: int) -> Dict[str, Any]:
    return dict(operation="number_is_greater_than", column=column, value=value)


def _test_render(
    arrow_table: pa.Table,
    params: Dict[str, Any],
    expected_table: Optional[pa.Table],
    expected_errors: List[I18nMessage] = [],
    *,
    settings: Settings = DefaultSettings()
):
    with tempfile.NamedTemporaryFile() as tf:
        path = Path(tf.name)
        actual_errors = render(arrow_table, params, path, settings=settings)
        if path.stat().st_size == 0:
            actual_table = None
        else:
            with pa.ipc.open_file(path) as f:
                actual_table = f.read_all()
        assert actual_errors == expected_errors
        if expected_table is None:
            assert actual_table is None
        else:
            assert actual_table is not None
            assert actual_table.column_names == expected_table.column_names
            for output_column, expected_column in zip(
                actual_table.itercolumns(), expected_table.itercolumns()
            ):
                assert output_column.type == expected_column.type
                assert output_column.to_pylist() == expected_column.to_pylist()


OUTPUT_COLUMN_TYPE = pa.dictionary(pa.int32(), pa.utf8())


def test_empty_table():
    _test_render(
        pa.table({"A": pa.array([], pa.int32())}),
        {"colname": "B", "labels": [{"value": "a", "condition": EQ("A", 1)}]},
        pa.table(
            {
                "A": pa.array([], pa.int32()),
                "B": pa.array([], OUTPUT_COLUMN_TYPE),
            }
        ),
    )


def test_ignore_null_condition():
    _test_render(
        pa.table({"A": [1, 2, 3]}),
        {"colname": "B", "labels": [{"value": "a", "condition": None}]},
        pa.table(
            {"A": [1, 2, 3], "B": pa.array([None, None, None], OUTPUT_COLUMN_TYPE)}
        ),
    )


def test_match_condition():
    _test_render(
        pa.table({"A": [1, 2, 3]}),
        {"colname": "B", "labels": [{"value": "a", "condition": EQ("A", 2)}]},
        pa.table(
            {"A": [1, 2, 3], "B": pa.array([None, "a", None]).dictionary_encode()}
        ),
    )


def test_duplicate_value_does_not_add_extra_dictionary_value():
    _test_render(
        pa.table({"A": [1, 2, 3]}),
        {
            "colname": "B",
            "labels": [
                {"value": "a", "condition": EQ("A", 2)},
                {"value": "a", "condition": EQ("A", 3)},
            ],
        },
        pa.table({"A": [1, 2, 3], "B": pa.array([None, "a", "a"]).dictionary_encode()}),
    )


def test_unused_value_does_not_add_extra_dictionary_value():
    _test_render(
        pa.table({"A": [1, 2, 3]}),
        {
            "colname": "B",
            "labels": [
                {"value": "a", "condition": EQ("A", 2)},
                {"value": "b", "condition": EQ("A", 4)},
            ],
        },
        pa.table(
            {"A": [1, 2, 3], "B": pa.array([None, "a", None]).dictionary_encode()}
        ),
    )


def test_double_match_first_takes_priority():
    _test_render(
        pa.table({"A": [1, 2, 3]}),
        {
            "colname": "B",
            "labels": [
                {"value": "a", "condition": EQ("A", 2)},
                {"value": "b", "condition": GT("A", 1)},
            ],
        },
        pa.table({"A": [1, 2, 3], "B": pa.array([None, "a", "b"]).dictionary_encode()}),
    )


def test_regex_errors():
    _test_render(
        pa.table({"A": ["a", "b"]}),
        {
            "colname": "B",
            "labels": [
                {
                    "value": "a",
                    "condition": dict(
                        operation="text_is",
                        column="A",
                        value="*",
                        isCaseSensitive=False,
                        isRegex=True,
                    ),
                },
                {
                    "value": "b",
                    "condition": dict(
                        operation="text_is",
                        column="A",
                        value="+",
                        isCaseSensitive=False,
                        isRegex=True,
                    ),
                },
            ],
        },
        None,
        [
            i18n_message(
                "error.invalidRegex",
                {"value": "*", "message": "no argument for repetition operator: *"},
            ),
            i18n_message(
                "error.invalidRegex",
                {"value": "+", "message": "no argument for repetition operator: +"},
            ),
        ],
    )


def test_overwrite_column():
    _test_render(
        pa.table({"A": [1, 2, 3], "B": [2, 3, 4]}),
        {"colname": "A", "labels": [{"value": "a", "condition": EQ("A", 2)}]},
        pa.table(
            {"A": pa.array([None, "a", None]).dictionary_encode(), "B": [2, 3, 4]}
        ),
    )


def test_empty_colname_is_no_op():
    _test_render(
        pa.table({"A": [1, 2, 3]}),
        {"colname": "", "labels": [{"value": "a", "condition": EQ("A", 2)}]},
        pa.table({"A": [1, 2, 3]}),
    )


def test_clean_colname():
    _test_render(
        pa.table({"A": [1, 2, 3]}),
        {"colname": "B\r", "labels": [{"value": "a", "condition": EQ("A", 2)}]},
        pa.table(
            {"A": [1, 2, 3], "B": pa.array([None, "a", None]).dictionary_encode()}
        ),
        [
            cjwmodule_i18n_message(
                "util.colnames.warnings.ascii_cleaned",
                {"n_columns": 1, "first_colname": "B"},
            )
        ],
    )
