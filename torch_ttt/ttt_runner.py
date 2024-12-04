import torch
from copy import deepcopy

def run_ttt(engine, inputs, optimizer_name="adam", num_steps=10, lr=1e-4, copy=False, engine_kwargs=None):

    running_engine = deepcopy(engine) if copy else engine

    engine_kwargs = engine_kwargs or {}

    running_engine.model.train()

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(running_engine.model.parameters(), lr=lr)

    for _ in range(num_steps):
        optimizer.zero_grad()
        _, loss = running_engine(inputs)
        loss.backward()
        optimizer.step()

    running_engine.model.eval()
    with torch.no_grad():
        final_outputs = running_engine.model(inputs)

    return final_outputs