import pytest
import torch
from torch import nn
from torch_ttt.engine_registry import EngineRegistry

# Minimal transformer-like model
class DummyTransformer(nn.Module):
    def __init__(self, vocab_size=100, hidden_dim=16, class_num=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.Linear(hidden_dim, vocab_size)  # will hook into this
        self.classifier = nn.Linear(vocab_size, class_num)
        self.class_num = class_num

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.encoder(x)  # This is the features_layer_name
        logits = self.classifier(x)
        return logits

@pytest.fixture
def dummy_input():
    return {"input_ids": torch.randint(0, 20, (4, 8))}    # (batch_size=4, seq_len=8)

@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
class TestMaskedTTTEngine:
    def setup_method(self):
        self.vocab_size = 20
        self.mask_token_id = 0
        self.class_num = 2
        self.features_layer_name = "encoder"

    def test_train_mode_forward(self, dummy_input, device):
        model = DummyTransformer(
            vocab_size=self.vocab_size,
            class_num=self.class_num
        ).to(device)
        engine = EngineRegistry.get_engine("masked_ttt")(
            model=model,
            mask_token_id=self.mask_token_id,
            features_layer_name=self.features_layer_name
        )
        engine.train()
        outputs, loss = engine({k: v.to(device) for k, v in dummy_input.items()})
        assert loss.requires_grad
        assert outputs.shape == (4, 8, self.class_num)

    def test_eval_mode_forward(self, dummy_input, device):
        model = DummyTransformer(
            vocab_size=self.vocab_size,
            class_num=self.class_num
        ).to(device)
        engine = EngineRegistry.get_engine("masked_ttt")(
            model=model,
            mask_token_id=self.mask_token_id,
            features_layer_name=self.features_layer_name
        )
        engine.eval()
        outputs, _ = engine({k: v.to(device) for k, v in dummy_input.items()})
        assert outputs.shape == (4, 8, self.class_num)

    def test_backward_pass(self, dummy_input, device):
        model = DummyTransformer(
            vocab_size=self.vocab_size,
            class_num=self.class_num
        ).to(device)
        engine = EngineRegistry.get_engine("masked_ttt")(
            model=model,
            mask_token_id=self.mask_token_id,
            features_layer_name=self.features_layer_name,
        )
        engine.train()
        optimizer = torch.optim.Adam(engine.parameters(), lr=1e-4)
        _, loss = engine({k: v.to(device) for k, v in dummy_input.items()})
        loss.backward()
        optimizer.step()
