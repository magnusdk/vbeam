from spekk import ops


def coherence_factor(beamformed_data: ops.array, receivers_axis: int):
    coherent = ops.abs(ops.sum(beamformed_data, receivers_axis)) ** 2
    incoherent = ops.sum(ops.abs(beamformed_data) ** 2, receivers_axis)
    # Don't divide by 0!
    incoherent = ops.where(incoherent == 0, 1, incoherent)
    return coherent / incoherent
