"""Numpy-like API for multiple backends.

This module contains an abstraction over the operations needed by the vbeam project, 
that can be implemented by multiple backends. Current included implementations are 
Tensorflow, JAX, and Numpy. You may register custom backends using the register_backend
function.

fastmath uses a "BYOD" principle: "Bring Your Own Dependency". If you only want
to use Tensorflow, then you only need to install Tensorflow in your project (and not 
JAX, for example, or any other potential backend). Setting the active_backend to your
preferred backend should (hopefully) let you use vbeam as if it is written in that
backend.

fastmath is considered a separate, but internal project, meaning that you should not 
depend on fastmath in your own projects (besides setting the active backend). This is to 
ease the maintenance of the fastmath module, and only implement the operations needed 
internally by vbeam.

fastmath is inspired by Google's deep learning library TRAX:
https://github.com/google/trax/tree/master/trax/fastmath


The API can be imported via:
>>> from vbeam.fastmath import backend_manager

You may then set your preferred backend (default is "numpy") on the backend_manager:
>>> backend_manager.active_backend = "jax"  # ...or something else

You may also set the backend using a context manager:
>>> with backend_manager.using_backend("tensorflow"):
...     print(backend_manager.active_backend)

Note that setting the backend sets it globally and is not thread-safe.
"""

import contextlib
from typing import Callable, Dict, Optional

from vbeam.fastmath.backend import Backend
from vbeam.fastmath.included_backends import (
    get_best_available_backend,
    included_backends,
)

_registered_backends: Dict[str, Callable[[], Backend]] = included_backends


def register_backend(name: str, get_backend: Callable[[], Backend]):
    """Register a new Backend.

    Args:
      name: The unique name of the backend, e.g. "my_new_backend".
      get_backend: A function that takes no arguments and returns the Backend. It
        typically also dynamically loads the needed modules.

    Example:
    # In file my_project/my_backend.py
    class MyBackend(Backend):
        ...  # Implementations of all methods

    # In file my_project/__init__.py
    def get_my_backend():
        from my_project.my_backend import MyBackend
        return MyBackend()

    register_backend("my_backend", get_my_backend)

    # You may now use your custom backend:
    backend_manager.active_backend = "my_backend"
    """
    if name in _registered_backends:
        raise ValueError(
            f'The name of the backend must be unique: A backend with the name "{name}" has already been registered.'
        )
    _registered_backends[name] = get_backend


class BackendManager:
    _active_backend: Optional[str]
    _backends_cache: Dict[str, Backend] = {}

    def __init__(self, initial_active_backend: Optional[str] = None):
        self._active_backend = initial_active_backend

    @property
    def active_backend(self) -> Backend:
        if self._active_backend is None:
            self._active_backend = get_best_available_backend()

        backend = self._backends_cache.get(self._active_backend, None)
        if backend is None:
            get_backend = _registered_backends[self._active_backend]
            backend = get_backend()
            self._backends_cache[self._active_backend] = backend

        return backend

    @active_backend.setter
    def active_backend(self, new_backend: str):
        self._active_backend = new_backend
        self.active_backend  # Eagerly instantiate backend

    @contextlib.contextmanager
    def using_backend(self, new_backend: str):
        prev_backend = self._active_backend
        self._active_backend = new_backend
        try:
            yield
        finally:
            self._active_backend = prev_backend


def proxy_backend(backend_manager: BackendManager) -> Backend:
    """Return a proxy-backend that forwards all calls to the active backend."""

    class ProxyBackend(Backend):
        def __getattribute__(self, attr):
            return getattr(backend_manager.active_backend, attr)

    return ProxyBackend()


backend_manager = BackendManager()
numpy = proxy_backend(backend_manager)
