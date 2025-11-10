import torch 
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from models.arqen_brain_v1 import ArqenBrain

df = pd.read_csv('data/arqen_data.csv')
X = df[["project_size", "floors", "region_code", "complexity", "duration_months"]].values
y = df["cost"].values.reshape(-1, 1)

scaler_x = StandardScaler()
scaler_y = StandardScaler()
X = torch.tensor(scaler_x.fit_transform(X), dtype=torch.float32)
y = torch.tensor(scaler_y.fit_transform(y), dtype=torch.float32)

input_dim = X.shape[1]
model = ArqenBrain(input_dim, hidden_size=64, output_size=1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(500):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/500], Loss: {loss.item():.4f}')



torch.save({
    'model_state': model.state_dict(),
    'scaler_x': scaler_x,
    'scaler_y': scaler_y
}, 'arqen_model.pth')

print("Arqen Brain trained and saved successfully.")