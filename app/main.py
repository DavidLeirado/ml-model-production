# Libraries
from fastapi import FastAPI
import numpy as np
import pickle
from pydantic import BaseModel


# Defining data model
class Data(BaseModel):
    x1: float
    x2: float


# Uploading ml model
filename = './model.sav'
model = pickle.load(open(filename, 'rb'))
loaded_model = pickle.load(open(filename, 'rb'))

# Starting fastapi app
app = FastAPI()


# Defining an endpoint. Expecting a json like {"x1": 0.0, "x2": 0.0}
@app.post("/model")
async def use_model(dt: Data):
    # Preparing data for the model
    data = np.array([dt.x1, dt.x2])
    data = data.reshape(1, -1)

    result = model.predict(data)

    # returning response
    return {"result": result[0]}

# To install fastapi: pip install fastapi[all]
# To initiate the app: uvicorn main:app --reload
