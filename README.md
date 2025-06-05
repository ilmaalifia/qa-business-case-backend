# QA System for Business Case

> This code is part of master thesis titled "Accelerating Business Case Development with Context-Aware AI: A Question Answering System Integrating PDF Corpora and Web Retrieval"

`qa-system-for-business-case` is a question answering system designed to enhance the business case development by accelerating knowledge and data acquisition process. It provides answers along with supporting references to help users make informed decisions. Built using [LangGraph](https://langchain-ai.github.io/langgraph/).

## 🖥️ Tested Machine Specs

This project has been tested on the following systems:

| OS       | CPU      | RAM  | Python Version |
| -------- | -------- | ---- | -------------- |
| macOS 15 | Apple M2 | 8 GB | 3.13           |

## ⚙️ Setup Instructions

You can set up the environment using either **Conda/Miniconda** or **Python venv**.

### 📦 Option 1: Using Conda/Miniconda

1. Create and activate a new environment:

   ```bash
   conda create -n qa_system_env python=3.13 -y
   conda activate qa_system_env
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

### 🐍 Option 2: Using Python Virtualenv

1. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   .\venv\Scripts\activate    # Windows
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

### 🔐 Setup Credential File and Update USER_AGENT

1. Copy the example file `.env.example` as `.env`:

```bash
cp .env.example .env          # macOS/Linux
copy .env.example .env        # Windows
```

2. Open the `.env` file in a text editor and fill in the required values, for example:

```env
...
MILVUS_URI=your-milvus-uri
MILVUS_TOKEN=your-milvus-token
...
```

3. Save the file. The system will automatically load variables from `.env` during execution.

## 🚀 How to Run

Run the the code using dev mode to enable LangSmith UI using the following command:

```bash
langgraph dev
```

## 🧪 Running Tests

To ensure everything is working correctly, this project includes automated tests that can be run using [pytest](https://docs.pytest.org/en/stable/).

1. Make sure your environment is set up and dependencies are installed (see Setup Instructions).

2. Run the tests from the project root directory:

```bash
pytest
```
