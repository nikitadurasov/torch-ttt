import torch.nn.functional as F
from torch_ttt.core.losses.base_loss import BaseLoss
from torch_ttt.core.loss_registry import LossRegistry

@LossRegistry.register("mean")
class MeanLoss(BaseLoss):

    def __init__(self):
        super().__init__()

    def __call__(self, model, inputs):
        outputs = model(inputs)
        return outputs.mean()