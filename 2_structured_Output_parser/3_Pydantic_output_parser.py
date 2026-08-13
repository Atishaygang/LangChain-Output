from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
load_dotenv()
import os
from langchain_core.prompts import PromptTemplate , load_prompt
from langchain_core.output_parsers import StrOutputParser , JsonOutputParser ,PydanticOutputParser
from pydantic import BaseModel , Field

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task="text-generation",
    temperature= 0.7,
    huggingfacehub_api_token= os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm = llm)

class Person(BaseModel):
    name: str = Field(description='Name of the character')
    age: int = Field(description="Age of the character")
    city: str = Field(description="City of the character")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template=('Write the name of a character who belongs from America and famous , {City} of the character and age of that character \n {format_instruction}'),
    input_variables=[
        'City'
    ],
    partial_variables={'format_instruction' : parser.get_format_instructions()}
)

prompt = template.format(City = 'America')

result = model.invoke(prompt)
final_result = parser.parse(result.content)
print(final_result)
