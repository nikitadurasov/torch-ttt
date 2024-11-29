import torch
from torch_ttt.core.loss_registry import LossRegistry

def run_ttt(model, inputs, loss_name, optimizer_name="adam", num_steps=10, lr=1e-4, loss_kwargs=None):

    loss_kwargs = loss_kwargs or {}

    model.train()
    loss_fn = LossRegistry.get_loss(loss_name)(**loss_kwargs) 

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(num_steps):
        optimizer.zero_grad()
        loss = loss_fn(model, inputs)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_outputs = model(inputs)

    return final_outputs