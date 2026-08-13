from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
load_dotenv()
import os
from langchain_core.prompts import PromptTemplate , load_prompt
from langchain_core.output_parsers import StrOutputParser

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task="text-generation",
    temperature= 0.7,
    huggingfacehub_api_token= os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm = llm)

template1 =PromptTemplate(
    template = 'Explain the whole {topic} very briefly ',
    input_variables=['topic']
)



template2 = PromptTemplate(
    template= 'Condensed it in 5 line summary. \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser |template2 | model | parser

result = chain.invoke({'topic': 'Neural Network'})

print(result)