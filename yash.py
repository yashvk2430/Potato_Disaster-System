from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import os
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "model_v2.h5")

MODEL = tf.keras.models.load_model(MODEL_PATH)


app = FastAPI()
CLASS_NAMES=["Early Blight", "Late Blight", "Healthy"]
@app.get("/ping")
async def ping():
    return {"message": "king"}

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    image = read_file_as_image(data)

    img_batch = np.expand_dims(image, axis=0)

    predictions = MODEL.predict(img_batch)

    predicted_index = int(np.argmax(predictions[0]))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[0][predicted_index])

    return {
        "class": str(predicted_class),
        "confidence": confidence,
        "all_probabilities": predictions[0].astype(float).tolist()
    }

    
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
