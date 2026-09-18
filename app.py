from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib

app = FastAPI(title="Titanic Survival Predictor")

model = joblib.load("model.joblib")


class Passenger(BaseModel):
    pclass: int = Field(..., ge=1, le=3, description="Класс билета: 1, 2 или 3")
    sex: str = Field(..., description="'male' или 'female'")
    age: float = Field(..., ge=0, le=100, description="Возраст")
    sibsp: int = Field(0, ge=0, description="Число супругов/братьев-сестёр на борту")
    parch: int = Field(0, ge=0, description="Число родителей/детей на борту")
    fare: float = Field(..., ge=0, description="Стоимость билета")
    embarked: str = Field("S", description="Порт посадки: 'C', 'Q' или 'S'")


@app.get("/")
def root():
    return {"status": "ok", "message": "POST /predict с данными пассажира"}


@app.post("/predict")
def predict(passenger: Passenger):
    X = pd.DataFrame([passenger.dict()])
    proba = float(model.predict_proba(X)[0, 1])
    prediction = int(proba >= 0.5)
    return {
        "survival_probability": round(proba, 3),
        "prediction": "выжил бы" if prediction == 1 else "не выжил бы",
    }
