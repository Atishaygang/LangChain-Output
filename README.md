# 🧩 LangChain Output Parsers — Learning Notes

A hands-on LangChain project documenting my learning and experiments with **Output Parsers and structured LLM responses** using Hugging Face, Qwen3, Pydantic, JSON, and JSON Schema.

The focus of this stage is understanding how an LLM's free-form response can be converted into usable Python data, and how provider capabilities affect structured-output features.

---

## 🚀 What I Have Learned

- `StrOutputParser`
- `JsonOutputParser`
- `PydanticOutputParser`
- Pydantic `BaseModel`
- Pydantic `Field`
- `Optional`
- JSON
- JSON Schema
- `with_structured_output()`
- Format instructions
- Parsing and validating LLM responses
- Provider-specific structured-output limitations
- Hugging Face + Qwen3 integration
- Qwen3 thinking/non-thinking mode

> **Note:** I experimented with the `|` operator to connect components, but **Chains/LCEL are not counted as learned yet**. They will be studied separately.

---

# 1. 🔤 StrOutputParser

`StrOutputParser` is used when the model's response is expected to be normal text.

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

result = model.invoke("Explain neural networks briefly")
final_result = parser.parse(result.content)

print(final_result)
```

Basic flow:

```text
LLM response
     ↓
StrOutputParser
     ↓
String
```

---

# 2. 🧾 JsonOutputParser

`JsonOutputParser` is used when the model should return structured information in JSON format.

Conceptually:

```text
LLM
 ↓
JSON response
 ↓
JsonOutputParser
 ↓
Python dictionary
```

Example:

```json
{
  "name": "John",
  "age": 25,
  "city": "New York"
}
```

**JSON is a representation of structured data.**

It is different from a Pydantic object.

---

# 3. 🐍 Pydantic

Pydantic lets us define and validate the structure of data using Python classes.

```python
from pydantic import BaseModel, Field
from typing import Optional

class Review(BaseModel):
    summary: str = Field(description="A brief summary of the review")
    sentiment: Optional[str] = Field(
        default=None,
        description="Positive or negative"
    )
    key_features: list[str] = Field(
        description="Key features mentioned in the review"
    )
    cons: Optional[list[str]] = Field(
        default=None,
        description="Cons from the review, if any"
    )
    pros: Optional[list[str]] = Field(
        default=None,
        description="Pros from the review, if any"
    )
    name: Optional[str] = Field(
        default=None,
        description="Name of the reviewer, if mentioned"
    )
```

### `Field()`

Descriptions should be supplied explicitly:

```python
name: str = Field(description="Name of the character")
```

rather than:

```python
name: str = Field("Name of the character")
```

because the positional argument is interpreted as a default value.

---

# 4. 🧩 PydanticOutputParser

This was the main parser approach I practiced with the Hugging Face API.

```python
from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(
    pydantic_object=Review
)
```

The parser can generate format instructions:

```python
parser.get_format_instructions()
```

These instructions can be inserted into a prompt so the model knows the expected output structure.

Example:

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template=(
        "Extract structured information from this review.\n"
        "{format_instructions}\n\n"
        "Review:\n{review}"
    ),
    input_variables=["review"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)
```

Conceptually:

```text
Pydantic Model
      ↓
PydanticOutputParser
      ↓
Format Instructions
      ↓
Prompt
      ↓
LLM
      ↓
Model response
      ↓
PydanticOutputParser
      ↓
Pydantic object
```

After parsing:

```python
result = parser.parse(response.content)

print(result)
print(result.summary)
print(result.pros)
print(result.cons)
```

The final result is a **Pydantic object**, not simply a JSON string.

---

# 5. 🆚 Pydantic vs JSON vs JSON Schema

These concepts are related but different.

### Pydantic

Defines and validates a Python data structure:

```python
class Review(BaseModel):
    summary: str
    sentiment: Optional[str] = None
    key_features: list[str]
```

### JSON

Represents structured data:

```json
{
  "summary": "Excellent phone",
  "sentiment": "positive",
  "key_features": [
    "Camera",
    "Battery"
  ]
}
```

### JSON Schema

Describes what a valid JSON structure should look like.

Conceptually:

```text
Pydantic
   ↓
Defines Python structure

JSON Schema
   ↓
Describes expected JSON structure

JSON
   ↓
Represents actual structured data

PydanticOutputParser
   ↓
Parses / validates the LLM output
```

---

# 6. 🔗 `with_structured_output()`

I also experimented with:

```python
model.with_structured_output(...)
```

The standard Pydantic approach was:

```python
structured_model = model.with_structured_output(Review)
```

With the Hugging Face integration used in these experiments, this produced:

```text
NotImplementedError:
Pydantic schema is not supported for function calling
```

### Important lesson

This does **not** mean Pydantic itself cannot be used with Hugging Face.

It means that the structured-output/function-calling mechanism exposed by the particular LangChain Hugging Face integration/model setup did not support this Pydantic approach.

The same style of implementation worked smoothly when following an OpenAI-based implementation.

---

# 7. 📋 JSON Schema Experiment

I also tried defining the structure directly as JSON Schema:

```python
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
            "description": "Analyze if the review is positive or negative"
        },
        "key_features": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key features from the review"
        },
        "cons": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Cons from the review if they exist"
        },
        "pros": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Pros from the review if they exist"
        },
        "name": {
            "type": "string",
            "description": "Name of the reviewer"
        }
    },
    "required": ["summary", "key_features"]
}

structured_model = model.with_structured_output(json_schema)
```

With the Hugging Face/Qwen setup, this did **not** provide the reliable structured-output behavior expected.

This reinforced the lesson that:

> `with_structured_output()` is provider/integration dependent.

---

# 8. 🤖 Hugging Face + Qwen3

The experiments used:

```text
Qwen/Qwen3-32B
```

through:

```python
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
```

Example:

```python
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-32B",
    task="text-generation",
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    huggingfacehub_api_token=os.getenv(
        "HUGGINGFACEHUB_ACCESS_TOKEN"
    )
)

model = ChatHuggingFace(llm=llm)
```

---

# 9. 🧠 Qwen3 Thinking / Non-Thinking Mode

Qwen3 supports thinking mode and can generate content inside:

```text
<think>
...
</think>
```

For structured parsing, I experimented with disabling thinking:

```python
model_kwargs={
    "extra_body": {
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }
}
```

A helper can also remove remaining thinking blocks:

```python
import re

def strip_think(msg):
    msg.content = re.sub(
        r"<think>.*?</think>",
        "",
        msg.content,
        flags=re.DOTALL
    ).strip()
    return msg
```

---

# 10. 🐛 Problems I Faced

### Pydantic + `with_structured_output()`

```text
NotImplementedError:
Pydantic schema is not supported for function calling
```

### Invalid JSON

When the model returned normal explanatory text instead of valid JSON, parsing failed with errors such as:

```text
JSONDecodeError
```

and:

```text
OutputParserException:
Invalid json output
```

### Provider-specific parameters

I also encountered compatibility issues while experimenting with Qwen3 thinking-related request parameters.

These errors helped demonstrate that the model/provider integration matters, even when the LangChain API looks similar.

---

# 11. 💡 Main Lessons

### 1. Parsing and generation are different

An output parser receives the model's response and converts/validates it.

```text
LLM
 ↓
Raw response
 ↓
Parser
 ↓
Structured Python data
```

### 2. Pydantic is not JSON

A Pydantic object:

```python
Review(...)
```

is a Python data model.

JSON is a serialized representation of structured data.

### 3. JSON Schema is not JSON

JSON Schema describes the expected structure.

JSON contains the actual data.

### 4. Provider compatibility matters

A feature demonstrated with OpenAI may not behave identically with Hugging Face.

### 5. Debugging is part of learning

The most important practical lesson from this section:

> **Don't just fix the error. Understand why the error happened.**

---

# 12. 🧪 Small Experiment with Multiple Components

While learning `StrOutputParser`, I also experimented with connecting components using `|`:

```python
chain = template1 | model | parser | template2 | model | parser
```

This showed how an output parser can sit between model calls and how one component's output can become the next component's input.

However:

> **Chains / LCEL are NOT considered completed learning yet.**

This was only an experiment while learning Output Parsers. Chains/LCEL will be studied separately.

---

# 13. 📌 Current Learning Status

```text
PromptTemplate                 ✅
ChatPromptTemplate             ✅
Messages                       ✅
MessagesPlaceholder            ✅
Chat History                   ✅

StrOutputParser                ✅
JsonOutputParser               ✅
PydanticOutputParser           ✅
Pydantic BaseModel             ✅
Pydantic Field                 ✅
Optional                       ✅
JSON                           ✅
JSON Schema                    ✅
Structured Output              ✅
with_structured_output()       ✅ Experimented
Provider limitations           ✅
Qwen3 thinking/non-thinking    ✅

Chains / LCEL                  ⏳ Not studied yet
RAG                            ⏳
Vector Stores                  ⏳
Retrievers                     ⏳
Agents                         ⏳
```

---

# 🔐 Environment Variables

Create a `.env` file:

```text
HUGGINGFACEHUB_ACCESS_TOKEN=hf_your_token_here
```

Load it with:

```python
from dotenv import load_dotenv
import os

load_dotenv()

token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
```

Never commit `.env` or API keys to GitHub.

Recommended `.gitignore`:

```text
.env
venv/
__pycache__/
```

---

# 📂 Example Structure

```text
Output-Parser/
│
├── StrOutputParser/
├── JsonOutputParser/
├── PydanticOutputParser/
├── Structured_output/
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

---

# ▶️ Running the Experiments

Create a virtual environment:

```bash
python -m venv venv
```

Activate on Windows PowerShell:

```powershell
.\venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run an experiment:

```bash
python filename.py
```

---

# 🎯 Next Step

Output Parsers are now complete for this stage.

The next step is a **small project using the concepts already learned**, before moving on to Chains/LCEL.

### Planned Mini Project — AI Review Analyzer

Input:

```text
A product review
```

Output:

```text
Summary
Sentiment
Rating
Key Features
Pros
Cons
Recommendation
```

The first version will stay focused on the concepts already learned.

---

# 👨‍💻 Author

**Atishay Jain**

B.Sc. (Hons.) Computer Science & Data Analytics  
IIT Patna

⭐ Built while learning LangChain, LLMs and Generative AI.
