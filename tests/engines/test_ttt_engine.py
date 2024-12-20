import pytest
import torch
from torch_ttt.engine_registry import EngineRegistry

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.cv1 = torch.nn.Conv2d(1, 10, 3)
        self.fc1 = torch.nn.Linear(10, 10)
        self.fc2 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.cv1(x))
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

@pytest.fixture()
def feat_input() -> torch.Tensor:
    return torch.ones((10, 1, 10, 10))

class TestTTTEngine1D:
    """Test cases for the TTT engine with 1D features."""


    def test_inference_train(self, feat_input):
        model = MLP()
        ttt_engine = EngineRegistry.get_engine("ttt")(model, "fc1")
        ttt_engine.train()
        ttt_engine(feat_input)

    def test_inference_eval(self, feat_input):
        model = MLP()
        ttt_engine = EngineRegistry.get_engine("ttt")(model, "fc1")
        ttt_engine.eval()
        ttt_engine(feat_input)

    def test_backward(self, feat_input):
        model = MLP()
        ttt_engine = EngineRegistry.get_engine("ttt")(model, "fc1")
        optimizer = torch.optim.Adam(ttt_engine.parameters(), lr=1e-4)
        ttt_engine.train()
        _, loss_ttt = ttt_engine(feat_input)
        loss_ttt.backward()
        optimizer.step()

class TestTTTEngine2D:
    """Test cases for the TTT engine with 1D features."""
    def test_inference_train(self, feat_input):
        model = MLP()
        ttt_engine = EngineRegistry.get_engine("ttt")(model, "cv1")
        ttt_engine.train()
        ttt_engine(feat_input)

    def test_inference_eval(self, feat_input):
        model = MLP()
        ttt_engine = EngineRegistry.get_engine("ttt")(model, "cv1")
        ttt_engine.eval()
        ttt_engine(feat_input)

    def test_backward(self, feat_input):
        model = MLP()
        ttt_engine = EngineRegistry.get_engine("ttt")(model, "cv1")
        optimizer = torch.optim.Adam(ttt_engine.parameters(), lr=1e-4)
        ttt_engine.train()
        _, loss_ttt = ttt_engine(feat_input)
        loss_ttt.backward()
        optimizer.step()