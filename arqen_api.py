import torch 
import pandas as pd
import numpy as np                                  
from fastapi import FastAPI
from pydantic import BaseModel
from models.arqen_brain_v1 import ArqenBrain
from sklearn.preprocessing import StandardScaler

tags_metadata = [
    {"name": "Root", "description": "Health and status of the Arqen API"},
    {"name": "Prediction", "description": "Predict project cost using the trained Market Brain model"},
]

app = FastAPI(
    title="Arqen Market Brain API",
    version="1.0",
    openapi_tags=tags_metadata
)

torch.serialization.add_safe_globals([StandardScaler])
torch.serialization.add_safe_globals([np._core.multiarray.scalar])

checkpoint = torch.load("arqen_model.pth", map_location="cpu", weights_only=False)
model_state = checkpoint["model_state"]
scaler_x = checkpoint["scaler_x"]
scaler_y = checkpoint["scaler_y"]

input_dim = len(scaler_x.mean_)
model = ArqenBrain(input_dim, hidden_size=64, output_size=1)   # ✅ fixed line
model.load_state_dict(model_state)
model.eval()

@app.get("/", tags=["Root"])
def home():
    return {"message": "Arqen Market Brain API is running ✅"}

class ProjectData(BaseModel):
    project_size: float
    floors: int
    region_code: int
    complexity: float
    duration_months: float

    class Config:
        schema_extra = {
            "example": {
                "project_size": 12000,
                "floors": 10,
                "region_code": 2,
                "complexity": 0.7,
                "duration_months": 14
            }
        }

@app.post("/predict", tags=["Prediction"])
def predict(data: ProjectData):
    df = pd.DataFrame([data.dict()])
    X = torch.tensor(scaler_x.transform(df), dtype=torch.float32)
    with torch.no_grad():
        y_pred = model(X)
    cost = scaler_y.inverse_transform(y_pred.numpy())[0][0]
    return {"predicted_cost_usd_million": round(float(cost), 2)}
