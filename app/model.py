import os
import torch
import wandb
from loadotenv import load_env
from torchvision.models import resnet18, ResNet
from torch import nn #can write nn.Linear instead of torch.nn.Linear
from pathlib import Path
from torchvision.transforms import v2 as transforms

MODELS_DIR = "models"
MODEL_FILENAME = "best_model.pth"

os.makedirs(MODELS_DIR, exist_ok=True) #will create a model dir if it not exist

def download_artifact():
    load_env() # this loads the variables in the .env file into the environment variables
    assert 'WANDB_API_KEY' in os.environ, "WANDB_API_KEY not found in environment variables"
    
    wandb.login(key=os.getenv('WANDB_API_KEY'))    
    api = wandb.Api()

    artifact_path = f"{os.getenv('WANDB_ORG')}/{os.getenv('WANDB_PROJECT')}/{os.getenv('WANDB_MODEL_NAME')}:{os.getenv('WANDB_MODEL_VERSION')}"
    artifact = api.artifact(artifact_path, type='model')
    artifact.download(root=MODELS_DIR)


def get_raw_model() -> ResNet: #returns18 model with the final layer changed to match the number of classes in our dataset
    architecture = resnet18(weights=None) 
    architecture.fc = nn.Sequential(
        nn.Linear(in_features=512, out_features=512),
        nn.ReLU(),
        nn.Linear(in_features=512, out_features=6) #change the last layer to match the number of classes in our dataset. out_features is the number of classes in our dataset
    )
    return architecture


def load_model() -> ResNet:
    '''Downloads the model artifact from Weights & Biases and loads the model weights into a ResNet18 architecture.'''
    download_artifact()
    model_path = Path(MODELS_DIR) / MODEL_FILENAME
    model = get_raw_model()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval() # Set the model to evaluation mode
    return model

def load_transforms():
    '''Returns the transformations used for preprocessing the input images.'''
    return transforms.Compose([
        transforms.Resize((256, 256)), # Resize the image to the size expected by ResNet18
        transforms.CenterCrop(224), # Center crop the image to the size expected by ResNet18
        transforms.ToImage(), # Convert the image to a PyTorch tensor
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize the image using the mean and std of the ImageNet dataset
    ])
#load_transformers = load_transformers() #call the function to get the transformers and store it in a variable

#print(load_transformers) #test the function by loading the model and printing it
#download_artifact()

