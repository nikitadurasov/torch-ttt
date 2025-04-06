import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
import math
from einops import rearrange

from torch_ttt.engine.base_engine import BaseEngine
from torch_ttt.engine_registry import EngineRegistry

__all__ = ["DeYOEngine"]

@EngineRegistry.register("deyo")
class DeYOEngine(BaseEngine):
    """**DeYO**: Destroy Your Object – Test-Time Adaptation with PLPD.

    Reference:
        "Entropy is Not Enough for Test-Time Adaptation" (ICLR 2024)
        Lee et al.
    """

    def __init__(
        self,
        model: nn.Module,
        optimization_parameters: Optional[Dict[str, Any]] = None,
        e_margin: float = 0.5 * math.log(1000),
        plpd_thresh: float = 0.2,
        ent_norm: float = 0.4 * math.log(1000),
        patch_len: int = 4,
        reweight_ent: float = 1.0,
        reweight_plpd: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.model.train()
        self._configure_norm_layers()

        self.optimization_parameters = optimization_parameters or {}
        self.e_margin = e_margin
        self.plpd_thresh = plpd_thresh
        self.ent_norm = ent_norm
        self.patch_len = patch_len
        self.reweight_ent = reweight_ent
        self.reweight_plpd = reweight_plpd

    def _configure_norm_layers(self):
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                for p in m.parameters():
                    p.requires_grad = True
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
            else:
                for p in m.parameters(recurse=False):
                    p.requires_grad = False

    def ttt_forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(inputs)
        entropy = self._softmax_entropy(outputs)

        # First filtering by entropy
        keep_mask = entropy < self.e_margin
        if keep_mask.sum() == 0:
            return outputs, entropy.mean() * 0

        x_sel = inputs[keep_mask].detach()
        outputs_sel = outputs[keep_mask]
        probs_sel = F.softmax(outputs_sel, dim=1)
        pseudo_labels = probs_sel.argmax(dim=1)

        # Patch shuffle for PLPD
        x_shuffled = self._patch_shuffle(x_sel)
        with torch.no_grad():
            outputs_shuffled = self.model(x_shuffled)
        probs_shuffled = F.softmax(outputs_shuffled, dim=1)

        # Compute PLPD
        plpd = (
            torch.gather(probs_sel, 1, pseudo_labels.unsqueeze(1)) -
            torch.gather(probs_shuffled, 1, pseudo_labels.unsqueeze(1))
        ).squeeze(1)

        # Second filtering by PLPD
        plpd_mask = plpd > self.plpd_thresh
        if plpd_mask.sum() == 0:
            return outputs, entropy.mean() * 0

        entropy_final = entropy[keep_mask][plpd_mask]
        plpd_final = plpd[plpd_mask]

        # Sample reweighting
        weight = (
            self.reweight_ent * (1 / torch.exp(entropy_final - self.ent_norm)) +
            self.reweight_plpd * (1 / torch.exp(-plpd_final))
        )
        loss = (entropy_final * weight).mean()

        return outputs, loss

    def _softmax_entropy(self, x: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(x, dim=1)
        log_probs = F.log_softmax(x, dim=1)
        return -(probs * log_probs).sum(dim=1)

    def _patch_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        patch_len = self.patch_len
        h_p, w_p = H // patch_len, W // patch_len

        # Resize to fit grid size
        resized = F.interpolate(x, size=(h_p * patch_len, w_p * patch_len), mode="bilinear", align_corners=False)
        patches = rearrange(resized, 'b c (ph h) (pw w) -> b (ph pw) c h w', ph=patch_len, pw=patch_len)
        
        # Shuffle patches per sample
        idx = torch.argsort(torch.rand(B, patches.size(1), device=x.device), dim=-1)
        patches = patches[torch.arange(B).unsqueeze(1), idx]
        
        shuffled = rearrange(patches, 'b (ph pw) c h w -> b c (ph h) (pw w)', ph=patch_len, pw=patch_len)
        return F.interpolate(shuffled, size=(H, W), mode="bilinear", align_corners=False)
