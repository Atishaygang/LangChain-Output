# 🧠 Structured Review Analyzer with LangChain

A simple LangChain project that uses an LLM to analyze product reviews and return the result in a **structured format**.

Instead of relying on the model to return plain text, this project defines a schema using Python's `TypedDict` and `Annotated` types and uses LangChain's `with_structured_output()` to generate structured data.

## 🚀 What This Project Does

The application takes a product review and extracts:

- 📝 Review summary
- 😊 Sentiment — positive or negative
- ⭐ Key features mentioned
- ✅ Pros
- ❌ Cons
- 👤 Reviewer's name, if mentioned

## 🛠️ Tech Stack

- Python
- LangChain
- Hugging Face
- Qwen 2.5 7B Instruct
- Python `TypedDict`
- `Annotated`
- `Optional`

## 🧩 Structured Output

The project defines a structured schema using `TypedDict`:

```python
class review(TypedDict):
    summary: Annotated[str, "Explain brief about the review"]
    sentiment: Annotated[str, "Classify as negative or positive"]
    key_features: Annotated[list[str], "Mention all features"]
    cons: Annotated[Optional[list[str]], "Mention all the cons from the review"]
    pros: Annotated[Optional[list[str]], "Mention all the pros from the review"]
    name: Annotated[
        Optional[str],
        "Name of the person who wrote it. Don't write name if it is not mentioned in review"
    ]
```

The schema tells the model what information should be returned and what each field represents.

## 🔗 LangChain Structured Output

The model is converted into a structured-output model using:

```python
structured_model = model.with_structured_output(review)
```

The review can then be passed directly to:

```python
result = structured_model.invoke(review_text)
```

Instead of receiving an unstructured response, the result follows the defined schema.

## 🤗 Hugging Face + Qwen

The project uses:

```text
Qwen/Qwen2.5-7B-Instruct
```

through LangChain's Hugging Face integration.

```python
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.5,
    huggingfacehub_api_token=os.getenv(
        "HUGGINGFACEHUB_ACCESS_TOKEN"
    )
)

model = ChatHuggingFace(llm=llm)
```

## 🔐 Environment Variables

Create a `.env` file in the project directory:

```text
HUGGINGFACEHUB_ACCESS_TOKEN=hf_your_token_here
```

The API key is loaded using:

```python
from dotenv import load_dotenv
import os

load_dotenv()

token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
```

### ⚠️ Important

Never commit your `.env` file or API key to GitHub.

Add the following to `.gitignore`:

```text
.env
venv/
__pycache__/
```

## 📂 Project Structure

```text
Structured-Output/
│
├── structured_output.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## ▶️ How to Run

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

Create a `.env` file:

```text
HUGGINGFACEHUB_ACCESS_TOKEN=hf_xxxxxxxxx
```

### 6. Run

```bash
python structured_output.py
```

## 📚 What I Learned

This project helped me understand:

- Structured LLM output
- `TypedDict`
- `Annotated`
- `Optional`
- LangChain `with_structured_output()`
- Defining a schema for LLM responses
- Extracting information from unstructured text
- Hugging Face LLM integration
- Qwen 2.5
- Environment variables and API key management

## 💡 Why Structured Output?

A normal LLM response might look like:

```text
The phone is excellent overall...
```

This is useful for humans, but applications often need predictable fields.

Structured output allows us to move toward:

```text
Review
 ├── Summary
 ├── Sentiment
 ├── Key Features
 ├── Pros
 ├── Cons
 └── Name
```

This makes LLM responses easier to use inside real applications, databases, APIs, and downstream processing pipelines.

## 🔮 Future Improvements

- Analyze multiple reviews automatically
- Create a Streamlit interface
- Export results to JSON/CSV
- Add sentiment confidence
- Process reviews from a CSV file
- Build a product review dashboard
- Combine structured output with RAG

## 👨‍💻 Author

**Atishay Jain**

B.Sc. (Hons.) Computer Science & Data Analytics  
IIT Patna

---

⭐ Built while learning LangChain and Generative AI.
