from langchain_huggingface import HuggingFaceEndpoint ,ChatHuggingFace
from dotenv import load_dotenv
import os
from typing import TypedDict , Optional , Annotated 
from pydantic import BaseModel
load_dotenv

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature= 0.5,
    huggingfacehub_api_token= os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

)

model =ChatHuggingFace(llm = llm)

