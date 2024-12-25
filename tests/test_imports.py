# ruff: noqa: F401
import unittest


class TestImports(unittest.TestCase):
    def test_import_loss_registry(self):
        try:
            from torch_ttt.loss_registry import LossRegistry
        except ImportError as e:
            self.fail(f"Failed to import LossRegistry: {e}")

    def test_import_engine_registry(self):
        try:
            from torch_ttt.engine_registry import EngineRegistry
        except ImportError as e:
            self.fail(f"Failed to import EngineRegistry: {e}")

    def test_import_base_loss(self):
        try:
            from torch_ttt.loss.base_loss import BaseLoss
        except ImportError as e:
            self.fail(f"Failed to import BaseLoss: {e}")

    def test_import_base_engine(self):
        try:
            from torch_ttt.engine.base_engine import BaseEngine
        except ImportError as e:
            self.fail(f"Failed to import BaseEngine: {e}")


if __name__ == "__main__":
    unittest.main()
