import unittest

class TestImports(unittest.TestCase):
    def test_import_loss_registry(self):
        try:
            from torch_ttt.core.loss_registry import LossRegistry
        except ImportError as e:
            self.fail(f"Failed to import LossRegistry: {e}")

    def test_import_ttt_runner(self):
        try:
            from torch_ttt.core.ttt_runner import run_ttt
        except ImportError as e:
            self.fail(f"Failed to import run_ttt: {e}")

    def test_import_base_loss(self):
        try:
            from torch_ttt.core.losses.base_loss import BaseLoss
        except ImportError as e:
            self.fail(f"Failed to import BaseLoss: {e}")

if __name__ == "__main__":
    unittest.main()