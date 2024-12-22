from abc import ABC, abstractmethod
from typing import Tuple
import torch.nn as nn
import torch
from copy import deepcopy

class BaseEngine(nn.Module, ABC):

    def __init__(self):
        nn.Module.__init__(self)
        self.optimization_parameters = {}

    @abstractmethod
    def ttt_forward(self, inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def forward(self, inputs):

        if self.training:
            return self.ttt_forward(inputs)

        # TODO: optimization pipeline should be more flexible and 
        # user-defined, need some special structure for that
        optimization_parameters = self.optimization_parameters or {} 

        optimizer_name = optimization_parameters.get("optimizer_name", "adam")
        num_steps = optimization_parameters.get("num_steps", 3)
        lr = optimization_parameters.get("lr", 1e-2)
        copy_model = optimization_parameters.get("copy_model", "True")

        running_engine = deepcopy(self) if copy_model else self

        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(running_engine.model.parameters(), lr=lr)

        loss = 0 # default value
        for _ in range(num_steps):
            optimizer.zero_grad()
            _, loss = running_engine.ttt_forward(inputs)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            final_outputs = running_engine.model(inputs)

        return final_outputs, loss
