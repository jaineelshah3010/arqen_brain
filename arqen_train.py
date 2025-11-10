import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from models.arqen_brain_v1 import ArqenBrain

# Load dataset
df = pd.read_csv('data/arqen_data.csv')

# Convert numeric columns safely
def to_float(x):
    try:
        return float(str(x).replace("Millions", "").replace(",", "").strip())
    except:
        return np.nan

numeric_cols = ["Store Sales", "Store Cost", "Store Area", "Grocery Area", "Frozen Area", "Meat Area", "Cost"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].apply(to_float)

df = df.fillna(0)

# Basic feature engineering
if "Store Sales" in df.columns and "Store Cost" in df.columns:
    df["Profit"] = df["Store Sales"] - df["Store Cost"]

# Prepare data
target_col = "Cost"
features = [c for c in df.columns if c != target_col and df[c].dtype != 'O']
X = df[features].values
y = df[[target_col]].values

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler_x, scaler_y = StandardScaler(), StandardScaler()
X_train = torch.tensor(scaler_x.fit_transform(X_train), dtype=torch.float32)
X_test = torch.tensor(scaler_x.transform(X_test), dtype=torch.float32)
y_train = torch.tensor(scaler_y.fit_transform(y_train), dtype=torch.float32)
y_test = torch.tensor(scaler_y.transform(y_test), dtype=torch.float32)

# Model setup
input_dim = X_train.shape[1]
model = ArqenBrain(input_dim, hidden_size=64, output_size=1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    pred = model(X_train)
    loss = loss_fn(pred, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

# Evaluate
model.eval()
with torch.no_grad():
    pred = model(X_test).numpy()
    act = y_test.numpy()

pred = scaler_y.inverse_transform(pred)
act = scaler_y.inverse_transform(act)

mae = mean_absolute_error(act, pred)
rmse = np.sqrt(mean_squared_error(act, pred))
r2 = r2_score(act, pred)

print("\n--- Model Evaluation ---")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# Calculate and print accuracy in percentage
mape = np.mean(np.abs((act - pred) / (np.abs(act) + 1e-8))) * 100
accuracy = max(0, 100 - mape)
print(f"Accuracy: {accuracy:.2f}%")