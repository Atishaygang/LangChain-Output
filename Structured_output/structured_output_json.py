from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
load_dotenv()
import os

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-32B",
    task="text-generation",
    temperature= 0.7,
    huggingfacehub_api_token= os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
    
    
)

model = ChatHuggingFace(llm = llm)

json_schema = {
  "title": "review_analysis",
  "type": "object",
  "properties": {
    "summary": {
      "type": "string",
      "description": "A brief summary of review"
    },
    "sentiment": {
      "type": "string",
      "description": "Analize if the review is positive or negative",
      "default": None
    },
    "key_features": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Key features from the review"
    },
    "cons": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "cons from the review if exist",
      "default": None
    },
    "pros": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "pros from the review if exist",
      "default": None
    },
    "name": {
      "type": "string",
      "description": "Name of the reviewer",
      "default": None
    }
  },
  "required": [
    "summary",
    "key_features"
  ]
}

structured_model = model.with_structured_output(json_schema)

result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3
processor makes everything lightning fast-whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily
lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me
away is the 200MP camera-the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x
actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with
bloatware-why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard
pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Cons:
Bulky and heavy-not great for one-handed use
Bloatware still exists in One UI
Expensive compared to competitors



""")

print(result)