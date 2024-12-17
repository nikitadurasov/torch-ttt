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
    return torch.ones((2, 1, 10, 10))

class TestTTTPPEngine:
    
    def test_simple_ttt_engine_inference(self, feat_input):
        model = MLP()
        ttt_engine = EngineRegistry.get_engine("ttt_pp")(model, "fc1")
        ttt_engine(feat_input)