import pytest
import torch
from torch_ttt.engine_registry import EngineRegistry


class CNNWithBN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(32, 5)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


@pytest.fixture()
def feat_input() -> torch.Tensor:
    return torch.ones((8, 1, 16, 16))


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
class TestTentEngine:
    """Test cases for the Tent engine."""

    def test_inference_train(self, feat_input, device):
        model = CNNWithBN().to(device)
        ttt_engine = EngineRegistry.get_engine("tent")(model)
        ttt_engine.train()
        outputs, loss_ttt = ttt_engine(feat_input.to(device))
        assert outputs.shape == (8, 5)
        assert loss_ttt.dim() == 0

    def test_inference_eval(self, feat_input, device):
        model = CNNWithBN().to(device)
        ttt_engine = EngineRegistry.get_engine("tent")(model)
        ttt_engine.eval()
        outputs, _ = ttt_engine(feat_input.to(device))
        assert outputs.shape == (8, 5)

    def test_backward(self, feat_input, device):
        model = CNNWithBN().to(device)
        ttt_engine = EngineRegistry.get_engine("tent")(model)
        optimizer = torch.optim.Adam(ttt_engine.parameters(), lr=1e-4)
        ttt_engine.train()
        _, loss_ttt = ttt_engine(feat_input.to(device))
        loss_ttt.backward()
        optimizer.step()

    def test_bn_params_trainable(self, feat_input, device):
        model = CNNWithBN().to(device)
        ttt_engine = EngineRegistry.get_engine("tent")(model)
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                for param in module.parameters():
                    assert param.requires_grad, f"BN param in {name} should be trainable"
        assert not model.fc.weight.requires_grad, "fc.weight should be frozen"

    def test_invalid_model_no_bn(self, feat_input, device):
        class LinearModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(256, 5)

            def forward(self, x):
                return self.fc1(x.view(x.size(0), -1))

        model = LinearModel().to(device)
        ttt_engine = EngineRegistry.get_engine("tent")(model)
        ttt_engine.train()
        outputs, _ = ttt_engine(feat_input.to(device))
        assert outputs.shape == (8, 5)

    def test_entropy_loss_positive(self, feat_input, device):
        model = CNNWithBN().to(device)
        ttt_engine = EngineRegistry.get_engine("tent")(model)
        ttt_engine.train()
        _, loss_ttt = ttt_engine(feat_input.to(device))
        assert loss_ttt.item() >= 0
