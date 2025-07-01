# QA System for Business Case

> This code is part of master thesis titled "Accelerating Business Case Development with Context-Aware AI: A Question Answering System Integrating PDF Corpora and Web Retrieval"

`qa-system-for-business-case` is a question answering system designed to enhance the business case development by accelerating knowledge and data acquisition process. It provides answers along with supporting references to help users make informed decisions. Built using [LangGraph](https://langchain-ai.github.io/langgraph/).

## 🖥️ Requirements

### ✅ Supported Systems

This project has been tested on the following system configuration:

| Operating System | Chip                              | RAM  | Python Version |
| ---------------- | --------------------------------- | ---- | -------------- |
| macOS 15         | Apple M2 (8-Core CPU, 8-Core GPU) | 8 GB | 3.13           |
| Windows 11       | Intel Core i7-1165G7 (4-Core CPU) | 16 GB | 3.13           |

> ℹ️ Other systems may work but have not been officially tested.

### 📦 External Dependencies

1. **Milvus** – Required if `MILVUS_ENABLE=true` in `.env` file for accessing preprocessed PDF document from [reverse-pdf-scraper](https://github.com/ilmaalifia/reverse-pdf-scraper).
2. **Tavily API Key** – Required if `TAVILY_ENABLE=true` in `.env` file for web search capabilities. You can obtain a key from Tavily.
3. **PubMed API Key** – Required if `PUBMED_ENABLE=true` in `.env` file for accessing PubMed scientific literature. You can register for an API key at NCBI.
4. **LLM API Key** – Required to access a language model (currently only supports OpenAI and DeepSeek). Make sure to configure your environment with the appropriate key for your provider.

## ⚙️ Setup Python Environment

You can set up the environment using either **Conda/Miniconda** or **Python venv**. The following guidelines uses **Miniconda**.

1. Install **Miniconda** using [this guidelines](https://www.anaconda.com/docs/getting-started/miniconda/install#basic-install-instructions).

2. Create and activate a new environment:

   ```bash
   conda create -n qa_system_env python=3.13 -y
   conda activate qa_system_env
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## 🔐 Setup Credential File and Update USER_AGENT

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

1. Make sure your environment is set up and dependencies are installed (see Setup Python Environment).

2. Run the tests from the project root directory:

```bash
pytest
```
