from spekk import ops


def normalized_decibels(data: ops.array):
    "Convert the data into decibels normalized for dynamic range."
    data_db = 20 * ops.nan_to_num(ops.log10(ops.abs(data)))
    return data_db - ops.max(data_db)
