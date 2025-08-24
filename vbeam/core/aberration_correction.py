from abc import abstractmethod
from spekk import Module, ops

class AberrationCorrection(Module):
    """An object that models the wavefront aberrations."""
    
    @abstractmethod
    def __call__(self) -> float:
        """Return the time delay aberrations (in seconds) 

        """

class NoAberrationCorrection(AberrationCorrection):

    def __call__(self) -> float:
        return 0.0
    
    