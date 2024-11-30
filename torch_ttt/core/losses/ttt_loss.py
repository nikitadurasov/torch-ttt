import torch.nn.functional as F
from torch_ttt.core.losses.base_loss import BaseLoss
from torch_ttt.core.loss_registry import LossRegistry

@LossRegistry.register("ttt")
class TTTLoss(BaseLoss):

    def __init__(self):
        super().__init__()

    def __call__(self, model, inputs):
        # TODO [P2]: check that model is TTTEngine
        _, loss = model(inputs)
        return loss