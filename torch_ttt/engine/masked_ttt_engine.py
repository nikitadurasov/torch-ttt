from torch_ttt.engine.base_engine import BaseEngine
from torch_ttt.engine_registry import EngineRegistry

__all__ = ["MaskedTTTEngine"]


# TODO: add cuda support
@EngineRegistry.register("masked_ttt")
class MaskedTTTEngine(BaseEngine):
    """Masked autoencoders-based **test-time training** approach."""

    def __init__(self):
        super().__init__()

    def __call__(self, inputs):
        pass
