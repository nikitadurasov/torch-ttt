import pytest
import torch
from torch_ttt.engine_registry import EngineRegistry


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.cv1 = torch.nn.Conv2d(1, 10, 3)
        self.bn1 = torch.nn.BatchNorm2d(10)
        self.fc1 = torch.nn.Linear(10, 10)
        self.fc2 = torch.nn.Linear(10, 5)

    def forward(self, x):
        x = self.bn1(torch.relu(self.cv1(x)))
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

@pytest.fixture()
def feat_input() -> torch.Tensor:
    return torch.ones((8, 1, 10, 10))

@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
class TestMemoEngine:
    """Test cases for the TTT engine with 1D features."""

    def test_inference_train(self, feat_input, device):
        model = MLP().to(device)
        ttt_engine = EngineRegistry.get_engine("deyo")(model)
        ttt_engine.train()
        ttt_engine(feat_input.to(device))

    def test_inference_eval(self, feat_input, device):
        model = MLP().to(device)
        ttt_engine = EngineRegistry.get_engine("deyo")(model)
        ttt_engine.eval()
        ttt_engine(feat_input.to(device))

    def test_backward(self, feat_input, device):
        model = MLP().to(device)
        ttt_engine = EngineRegistry.get_engine("deyo")(model)
        optimizer = torch.optim.Adam(ttt_engine.parameters(), lr=1e-4)
        ttt_engine.train()
        _, loss_ttt = ttt_engine(feat_input.to(device))
        loss_ttt.backward()
        optimizer.step()
