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


download_artifact()

