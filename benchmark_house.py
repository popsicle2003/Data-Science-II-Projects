
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def benchmark():
    # Simulate house dataset (800 rows, 7 features)
    X = torch.randn(800, 7)
    y = torch.randn(800)
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    model = nn.Sequential(
        nn.Linear(7, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    start_time = time.time()
    epochs = 300
    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb).squeeze()
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
    
    end_time = time.time()
    print(f"Time for 1 trial (300 epochs): {end_time - start_time:.2f}s")

if __name__ == "__main__":
    benchmark()
