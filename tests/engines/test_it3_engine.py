# tests/test_ittt_engine.py
import pytest
import torch
from torch import nn
from torch_ttt.engine_registry import EngineRegistry


# ----------------------------------------------------------------------
#  Réseaux jouets couvrant toutes les combinaisons
# ----------------------------------------------------------------------
class VectorNet(nn.Module):
    """Entrée 2-D → sortie 2-D"""
    def __init__(self, in_c=8, hid=16, out_c=4):
        super().__init__()
        self.fc1 = nn.Linear(in_c, hid)
        self.fc2 = nn.Linear(hid, out_c)

    def forward(self, x):          # x: (B, in_c)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class ConvFCNet(nn.Module):
    """Entrée 4-D → sortie 2-D ; injection sur conv"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.head = nn.Linear(16, 10)

    def forward(self, x):          # x: (B,3,H,W)
        f = torch.relu(self.conv(x))
        g = torch.nn.functional.adaptive_avg_pool2d(f, 1).flatten(1)
        return self.head(g)


class ConvSegNet(nn.Module):
    """Entrée 4-D → sortie 4-D ; injection sur conv"""
    def __init__(self, in_c=1, mid=8, out_c=1):
        super().__init__()
        self.enc = nn.Conv2d(in_c, mid, 3, padding=1)
        self.dec = nn.Conv2d(mid, out_c, 3, padding=1)

    def forward(self, x):          # x: (B,1,H,W)
        f = torch.relu(self.enc(x))
        return torch.sigmoid(self.dec(f))


# ----------------------------------------------------------------------
#  Fixtures d’entrées
# ----------------------------------------------------------------------
@pytest.fixture()
def vec_batch():
    return torch.ones((6, 8))          # (B=6, in_c=8)


@pytest.fixture()
def img_batch3c():
    return torch.ones((4, 3, 16, 16))  # (B,3,H,W)


@pytest.fixture()
def img_batch1c():
    return torch.ones((2, 1, 16, 16))  # (B,1,H,W)


# ----------------------------------------------------------------------
#  Paramètres CPU / CUDA
# ----------------------------------------------------------------------
DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


# ----------------------------------------------------------------------
#  1)  2-D → 2-D  (VectorNet)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("device", DEVICES)
def test_ittt_vectornet(vec_batch, device):
    model = VectorNet().to(device)
    engine = EngineRegistry.get_engine("ittt")(model, features_layer_name="fc1")
    engine.train()
    y, loss = engine(vec_batch.to(device))
    assert y.shape == (6, 4)
    assert loss.requires_grad


# ----------------------------------------------------------------------
#  2)  4-D → 2-D  (ConvFCNet)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("device", DEVICES)
def test_ittt_convfcnet(img_batch3c, device):
    model = ConvFCNet().to(device)
    engine = EngineRegistry.get_engine("ittt")(model, "conv")
    engine.train()
    y, loss = engine(img_batch3c.to(device))
    assert y.dim() == 2 and y.size(0) == 4
    assert loss.requires_grad


# ----------------------------------------------------------------------
#  3)  4-D → 4-D  (ConvSegNet)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("device", DEVICES)
def test_ittt_convsegnet(img_batch1c, device):
    model = ConvSegNet().to(device)
    engine = EngineRegistry.get_engine("ittt")(model, "enc")
    engine.eval()                                 # adaptation en mode eval
    y, loss = engine(img_batch1c.to(device))

    assert y.shape == (2, 1, 16, 16)


# ----------------------------------------------------------------------
#  4)  Couche inexistante  →  ValueError
# ----------------------------------------------------------------------
def test_ittt_invalid_layer():
    model = VectorNet()
    with pytest.raises(ValueError):
        EngineRegistry.get_engine("ittt")(model, "not_there")
