import iris
import torch

class ShmemCompat:
    """
    Compat layer mimicking the old `iris.Shmem` API.
    """

    def __init__(self, heap_size=2**30):
        # Iris() is the new SHMEM context class
        self.ctx = iris.Iris(heap_size=heap_size)

    def zeros(self, *args, **kwargs):
        # Old code passes device="cuda", but Iris zeros() doesn't accept it
        kwargs.pop("device", None)
        return self.ctx.zeros(*args, **kwargs)

    def get_heap_bases(self):
        return self.ctx.get_heap_bases()


    def __getattr__(self, name):
        return getattr(self.ctx, name)
