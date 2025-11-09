import torch
from arqen_brain.models.arqen_brain_v1 import ArqenBrain

checkpoint = torch.load("arqen_model.pth", map_location="cpu", weights_only=False)
print(checkpoint.keys())  # should show model_state, scaler_x, scaler_y

model = ArqenBrain(input_size=5, hidden_size=64, output_size=1)
model.load_state_dict(checkpoint["model_state"])
model.eval()
print("✅ Model loaded successfully!")