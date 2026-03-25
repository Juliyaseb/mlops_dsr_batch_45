from loadotenv import load_env
import os
load_env()  # Load environment variables from .env file
assert 'WANDB_API_KEY' in os.environ, "WANDB_API_KEY is not set in the environment variables"
import wandb

artifact_path = "juliyasebastian001-data-science-retreat/mlops_dsr_batch_45/resnet18:v0"

MODELS_DIR = "models"
MODEL_FILENAME = "best_model.pth"

os.makedirs(MODELS_DIR, exist_ok=True) #will create a model dir if it not exist

api=wandb.Api() #initialize the api

wandb.login(key=os.getenv('WANDB_API_KEY')) #login to wandb using the api key from env variable
artifact=api.artifact(artifact_path, type="model") #get the artifact from wandb
artifact.download(root=MODELS_DIR) #download the artifact and get the local path



