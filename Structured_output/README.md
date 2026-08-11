# 🧠 Structured Review Analyzer with LangChain

A LangChain project that explores **structured LLM output** for product-review analysis using **Hugging Face, Qwen3-32B, JSON Schema, and Pydantic**.

The project explores multiple structured-output approaches and documents the compatibility limitations encountered with the Hugging Face integration.

---

## 🚀 What This Project Does

The application takes a product review and extracts:

- 📝 Review summary
- 😊 Sentiment — positive or negative
- ⭐ Key features mentioned
- ✅ Pros
- ❌ Cons
- 👤 Reviewer's name, if mentioned

---

## 🛠️ Tech Stack

- Python
- LangChain
- LangChain Hugging Face integration
- Hugging Face Inference API
- Qwen3-32B
- Pydantic
- JSON Schema
- `PydanticOutputParser`
- `python-dotenv`

---

# 🧩 Structured Output

The intended output structure is defined using Pydantic:

```python
class Review(BaseModel):
    summary: str
    sentiment: Optional[str] = None
    key_features: list[str]
    cons: Optional[list[str]] = None
    pros: Optional[list[str]] = None
    name: Optional[str] = None
```

The schema tells the LLM what information should be extracted from an unstructured review.

---

# 🔗 Attempt 1 — `with_structured_output()` with Pydantic

The first approach followed the standard LangChain pattern:

```python
structured_model = model.with_structured_output(Review)
```

With the Hugging Face integration used in this project, this resulted in:

```text
NotImplementedError:
Pydantic schema is not supported for function calling
```

### Important Learning

This does **not** mean that Pydantic itself cannot be used with Hugging Face.

It means that the particular structured-output/function-calling mechanism used by the LangChain Hugging Face integration was not supported in this setup.

---

# 🧩 Attempt 2 — JSON Schema with `with_structured_output()`

The next approach was to define the structure directly as JSON Schema:

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
```

Then:

```python
structured_model = model.with_structured_output(json_schema)
```

This was also tested with the Hugging Face/Qwen setup.

### Result

The JSON Schema approach did not provide the reliable structured-output behavior expected from the provider integration.

This demonstrated an important point:

> `with_structured_output()` is provider/integration dependent. The same API can behave differently across different model providers.

---

# ✅ Working Approach — `PydanticOutputParser`

The working approach with the Hugging Face API was to explicitly use:

```python
PydanticOutputParser
```

Example:

```python
parser = PydanticOutputParser(
    pydantic_object=Review
)
```

The parser generates format instructions from the Pydantic schema:

```python
parser.get_format_instructions()
```

These instructions are included in the prompt so the model knows the expected structure.

The pipeline becomes:

```text
Pydantic Model
      ↓
PydanticOutputParser
      ↓
Format Instructions
      ↓
Prompt
      ↓
Hugging Face LLM
      ↓
Structured response
      ↓
PydanticOutputParser
      ↓
Validated Pydantic object
```

Example:

```python
chain = prompt | model | strip_think | parser

result = chain.invoke({
    "review": review_text
})
```

This approach does not rely on Hugging Face function calling.

---

# 🤖 Qwen3 Thinking Mode

The project uses:

```text
Qwen/Qwen3-32B
```

Qwen3 supports thinking mode, which can produce content inside:

```text
<think>
...
</think>
```

For this structured-output task, thinking was disabled through the Hugging Face request:

```python
model_kwargs={
    "extra_body": {
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }
}
```

A helper can also remove any remaining thinking blocks:

```python
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

# 🧠 Pydantic vs JSON vs JSON Schema

These concepts are related but are **not the same thing**.

### Pydantic

Pydantic defines and validates a Python data structure:

```python
class Review(BaseModel):
    summary: str
    sentiment: Optional[str] = None
    key_features: list[str]
    cons: Optional[list[str]] = None
    pros: Optional[list[str]] = None
    name: Optional[str] = None
```

### JSON

JSON is a machine-readable representation of structured data:

```json
{
  "summary": "Excellent flagship phone",
  "sentiment": "positive",
  "key_features": [
    "Snapdragon 8 Gen 3",
    "200MP camera"
  ],
  "pros": [
    "Excellent performance"
  ],
  "cons": [
    "Heavy",
    "Expensive"
  ],
  "name": null
}
```

### JSON Schema

JSON Schema describes what a valid JSON structure should look like.

Conceptually:

```text
Pydantic
   ↓
Defines the Python structure

JSON Schema
   ↓
Describes the expected JSON structure

JSON
   ↓
Represents the structured data

PydanticOutputParser
   ↓
Parses/validates the LLM output
```

---

# 🤗 Hugging Face + Qwen3

The project uses:

```text
Qwen/Qwen3-32B
```

through LangChain's Hugging Face integration.

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
    ),
    model_kwargs={
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }
    }
)

model = ChatHuggingFace(llm=llm)
```

---

# 🔐 Environment Variables

Create a `.env` file in the project directory:

```text
HUGGINGFACEHUB_ACCESS_TOKEN=hf_your_token_here
```

Load it using:

```python
from dotenv import load_dotenv
import os

load_dotenv()

token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
```

### ⚠️ Important

Never commit your `.env` file or API key to GitHub.

Recommended `.gitignore`:

```text
.env
venv/
__pycache__/
```

---

# 📂 Project Structure

```text
Structured-Output/
│
├── structured_output.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

---

# ▶️ How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Structured-Output
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Add your Hugging Face API key

Create `.env`:

```text
HUGGINGFACEHUB_ACCESS_TOKEN=hf_xxxxxxxxx
```

### 6. Run

```bash
python structured_output.py
```

---

# 📚 What I Learned

This project helped me understand:

- Structured LLM output
- Pydantic `BaseModel`
- Pydantic `Field`
- `Optional`
- JSON
- JSON Schema
- `PydanticOutputParser`
- `with_structured_output()`
- Provider-specific structured-output limitations
- Prompt-based structured extraction
- Hugging Face LLM integration
- Qwen3 thinking/non-thinking mode
- Parsing and validating LLM responses
- Environment variables and API key management

---

# ⚠️ Key Learning: Provider Compatibility Matters

One of the most useful lessons from this project was that **LangChain abstractions do not guarantee identical behavior across every LLM provider**.

The same:

```python
model.with_structured_output(...)
```

concept can behave differently depending on the underlying model/provider integration.

In this project, the Hugging Face setup did not provide the expected Pydantic/function-calling or JSON-Schema structured-output path.

The working approach was:

```text
Hugging Face
      ↓
Prompt + format instructions
      ↓
Qwen3
      ↓
PydanticOutputParser
      ↓
Validated Pydantic result
```

This was an important practical lesson beyond simply following a tutorial.

---

# 💡 Why Structured Output?

A normal LLM response might look like:

```text
The phone is excellent overall. It has great performance
and camera quality, but it is heavy and expensive.
```

This is useful for humans, but applications often need predictable fields.

Structured output allows us to work with:

```text
Review
 ├── Summary
 ├── Sentiment
 ├── Key Features
 ├── Pros
 ├── Cons
 └── Name
```

This makes LLM responses easier to use inside:

- APIs
- Databases
- Applications
- Data pipelines
- Downstream processing

---

# 🔮 Future Improvements

- Analyze multiple reviews automatically
- Create a Streamlit interface
- Export results to JSON/CSV
- Add sentiment confidence
- Process reviews from a CSV file
- Build a product review dashboard
- Combine structured output with RAG
- Experiment with providers that support native structured output

---

# 👨‍💻 Author

**Atishay Jain**

B.Sc. (Hons.) Computer Science & Data Analytics  
IIT Patna

---

⭐ Built while learning LangChain and Generative AI.
