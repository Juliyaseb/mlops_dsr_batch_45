from fastapi import FastAPI , File, UploadFile,Depends
import pydantic
from pydantic import BaseModel
import io
from PIL import Image
from app.model import load_model, load_transforms
from torchvision.models import resnet18, ResNet
import torch
from torchvision.transforms import v2 as transforms

categories = ['fresh_apple', 'fresh_banana', 'fresh_orange', 
              'rotten_apple', 'rotten_banana', 'rotten_orange']


class Result(BaseModel):
    category: str #the predicted category of input image
    confidence: float #the confidence score of the prediction between 0 and 1(a probability)

app = FastAPI()

@app.get("/")
def read_root():
    return {"Message":"""welcome to the image classification API. Use the /predict endpoint to get predictions from the model."""}

@app.post('/predict',response_model=Result)
async def predict(input_image:UploadFile = File(...),
                  model:ResNet=Depends(load_model),
                  transforms:transforms.Compose=Depends(load_transforms)
                  ) -> Result:
    
    image= Image.open(io.BytesIO(await input_image.read())).convert("RGB") 


    print(f'The image mode is this (DEBUGGING MSG){image.mode}')

    image=transforms(image).reshape(1, 3, 224, 224) #add a batch dimension to the image tensor
    
    model.eval() 
    
   
    
    with torch.inference_mode():
        logits=model(image)
        probs = torch.nn.functional.softmax(logits, dim=1) # Apply softmax to get probabilities
        confidence, predicted_class_idx = torch.max(probs, dim=1) # Get the predicted class and its confidence score
        predicted_category = categories[predicted_class_idx.item()] # Get the predicted category name from the
    
    result=Result(category=predicted_category, confidence=confidence.item())
    
    return result

