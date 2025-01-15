from spekk import ops


def coherence_factor(beamformed_data: ops.array, axis: int):
    coherent = ops.abs(ops.sum(beamformed_data, axis=axis)) ** 2
    incoherent = ops.sum(ops.abs(beamformed_data) ** 2, axis=axis)
    # Don't divide by 0!
    incoherent = ops.where(incoherent == 0, 1, incoherent)
    return coherent / incoherent
