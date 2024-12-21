import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
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

class TestTTTPPEngin1D:
    
    def test_inference_train(self, feat_input):
        model = MLP()
        ttt_engine = EngineRegistry.get_engine("ttt_pp")(model, "fc1")
        ttt_engine.train()
        ttt_engine(feat_input)

    def test_inference_eval_without_statistics(self, feat_input):
        model = MLP()
        ttt_engine = EngineRegistry.get_engine("ttt_pp")(model, "fc1")
        ttt_engine.eval()

        # Expect an exception when calling ttt_engine with feat_input
        with pytest.raises(ValueError) as excinfo:
            ttt_engine(feat_input)
        
        assert "Reference statistics are not computed. Please call `compute_statistics` method." in str(excinfo.value)

    def test_inference_eval_with_statistics(self, feat_input):
        model = MLP()
        ttt_engine = EngineRegistry.get_engine("ttt_pp")(model, "fc1")

        dataset = TensorDataset(feat_input)
        dataloader = DataLoader(dataset, batch_size=2)

        ttt_engine.compute_statistics(dataloader)
        ttt_engine.eval()
        ttt_engine(feat_input)

class TestTTTPPEngin2D:
    
    def test_inference_2d_exception(self, feat_input):
        model = MLP()

        # Expect an exception when calling ttt_engine with 2d features
        with pytest.raises(TypeError) as excinfo:
            ttt_engine = EngineRegistry.get_engine("ttt_pp")(model, "cv1")