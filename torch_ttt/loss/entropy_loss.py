import torch
import torch.nn.functional as F
from torch_ttt.loss.base_loss import BaseLoss
from torch_ttt.loss_registry import LossRegistry


@LossRegistry.register("entropy")
class EntropyLoss(BaseLoss):
    def __init__(self):
        super().__init__()

    def __call__(self, model, inputs):
        logits = model(inputs)
        probs = F.softmax(logits, dim=1)
        return -torch.sum(probs * torch.log(probs), dim=1).mean()
