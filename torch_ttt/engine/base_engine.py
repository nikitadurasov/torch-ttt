from abc import ABC, abstractmethod
from typing import Tuple
import torch.nn as nn
import torch
from copy import deepcopy


class BaseEngine(nn.Module, ABC):

    def __init__(self):
        nn.Module.__init__(self)
        self.optimization_parameters = {}
        self.optimizer = None

    @abstractmethod
    def ttt_forward(self, inputs, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def forward(self, inputs, **kwargs):

        if self.training:
            return self.ttt_forward(inputs, **kwargs)

        # TODO: optimization pipeline should be more flexible and
        # user-defined, need some special structure for that
        optimization_parameters = self.optimization_parameters or {}

        optimizer_name = optimization_parameters.get("optimizer_name", "adam")
        num_steps = optimization_parameters.get("num_steps", 3)
        lr = optimization_parameters.get("lr", 1e-2)
        copy_model = optimization_parameters.get("copy_model", "True")

        running_engine = deepcopy(self) if copy_model else self

        parameters = filter(lambda p: p.requires_grad, running_engine.model.parameters())
        if copy_model or self.optimizer is None:
            
            if optimizer_name == "adam":
                self.optimizer = torch.optim.Adam(parameters, lr=lr)

        loss = torch.Tensor([0.0]) # default value
        for i in range(num_steps):
            self.optimizer.zero_grad()
            _, loss = running_engine.ttt_forward(inputs, **kwargs)
            loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            if isinstance(inputs, dict):
                final_outputs = running_engine.model(**inputs)
            else:
                final_outputs = running_engine.model(inputs)

        return final_outputs, loss