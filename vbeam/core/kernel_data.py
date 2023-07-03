from typing import Sequence, TypeVar, Union

TSelf = TypeVar("TSelf", bound="KernelData")


class KernelData:
    """KernelData is the base class for containers of data that is needed by a kernel.

    KernelData provides methods that lets us treat it like a mapping, meaning that we
    can easily pass the whole kernel data object as kwargs to a kernel function,
    e.g.: signal_for_point(**kernel_data)."""

    # Defining keys and __getitem__ means this can be used as a mapping
    def keys(self) -> Sequence[str]:
        raise NotImplementedError

    def __getitem__(self, key: Union[str, int]):
        if isinstance(key, int):
            key = list(self.keys())[key]
        return getattr(self, key)

    def copy(self: TSelf) -> TSelf:
        raise NotImplementedError
