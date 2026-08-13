from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
load_dotenv()
import os
from langchain_core.prompts import PromptTemplate , load_prompt
from langchain_core.output_parsers import StrOutputParser , JsonOutputParser

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task="text-generation",
    temperature= 0.7,
    huggingfacehub_api_token= os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm = llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template=("Write the name , age and city of Doraemon's main character \n {format_instruction}"),
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

prompt = template.format()

result = model.invoke(prompt)
final_result = parser.parse(result.content)
print(final_result)
print(type(final_result))

