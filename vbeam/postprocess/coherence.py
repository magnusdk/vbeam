from spekk import ops


def coherence_factor(
    beamformed_data: ops.array,
    active_element_weights: ops.array,
    *,
    ord=2,
    axis: int,
):
    
    coherent   = ops.abs(ops.sum(beamformed_data, axis=axis)) ** ord
    incoherent = ops.sum(ops.abs(beamformed_data) ** ord, axis=axis)

    weight = active_element_weights
    if axis in weight.dims:
        weight = ops.sum(weight>0.0, axis=axis) ** (ord - 1)

    # Don't divide by 0!
    incoherent = ops.where(incoherent == 0, 1, incoherent)
    weight = ops.where(weight == 0, 1, weight)
    
    return coherent / incoherent / weight
    

