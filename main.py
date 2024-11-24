import os
from fastapi import FastAPI
from dotenv import load_dotenv
from langchain.smith.evaluation.runner_utils import logger
from langchain_huggingface import HuggingFaceEndpoint
from sympy.physics.units import temperature
from torch.backends.mkl import verbose

load_dotenv()
app = FastAPI()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
model_id = "microsoft/Phi-3-mini-4k-instruct"

@app.get("/")
def joke():
    llm = HuggingFaceEndpoint(
        repo_id=model_id,
        task="text-generation",
        max_new_tokens=512,
        repetition_penalty=2,
        token=hf_token
    )

    response = llm.invoke('Tell me a knock knock joke')
    return {"joke" : response}
