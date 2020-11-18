import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pyarrow as pa
from cjwmodule import i18n
from cjwmodule.arrow.condition import condition_to_mask
from cjwmodule.util.colnames import gen_unique_clean_colnames_and_warn

__all__ = ["render"]


def _generate_label_column(
    arrow_table: pa.Table, label_specs: List[Dict[str, Any]]
) -> Tuple[pa.DictionaryArray, List]:
    errors = []
    nulls = np.ones(len(arrow_table), np.bool_)  # start all-null
    values = []  # first value will have index=0
    indices = np.zeros(len(arrow_table), np.int32)  # start all-0 (but they're null)

    for label_spec in label_specs:
        value = label_spec["value"]
        condition = label_spec["condition"]

        if condition is None:
            continue

        try:
            mask = condition_to_mask(arrow_table, condition)
        except re.error as err:
            errors.append(
                i18n.trans(
                    "error.invalidRegex",
                    "Invalid regular expression “{pattern}”: {message}",
                    {"value": err.pattern, "message": err.msg},
                )
            )
            continue

        np_mask = mask.to_numpy() & nulls  # only overwrite nulls

        if not np.any(np_mask):
            # This label will apply to zero rows, because higher-priority labels
            # were already written and/or the condition didn't match.
            continue

        try:
            index = values.index(value)  # more rows for an existing label
        except ValueError:
            index = len(values)  # new label -- first one is index=0
            values.append(value)

        nulls[np_mask] = False
        if index > 0:  # cute, premature optimization: no need to set index=0
            indices[np_mask] = index

    return (
        pa.DictionaryArray.from_arrays(
            indices, pa.array(values, pa.utf8()), mask=nulls
        ),
        errors,
    )


def _add_column(
    arrow_table: pa.Table, name: str, data: pa.Array, *, settings
) -> [pa.Table, List[str]]:
    if name in arrow_table.column_names:
        return (
            arrow_table.set_column(arrow_table.column_names.index(name), name, data),
            [],
        )
    else:
        [clean_colname], errors = gen_unique_clean_colnames_and_warn(
            [name], existing_names=arrow_table.column_names, settings=settings
        )
        return arrow_table.append_column(clean_colname, data), errors


def render(arrow_table: pa.Table, params, output_path, *, settings, **kwargs):
    if not params["colname"]:
        with pa.ipc.RecordBatchFileWriter(output_path, arrow_table.schema) as writer:
            writer.write_table(arrow_table)
        return []  # no errors

    label_column, errors = _generate_label_column(arrow_table, params["labels"])

    if errors:
        return errors

    output_table, add_column_errors = _add_column(
        arrow_table, params["colname"], label_column, settings=settings
    )
    errors.extend(add_column_errors)

    with pa.ipc.RecordBatchFileWriter(output_path, output_table.schema) as writer:
        writer.write_table(output_table)
    return errors
