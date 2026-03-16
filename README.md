# Self-RAG System with LangGraph

This project implements an advanced Self-Reflective Retrieval-Augmented Generation (Self-RAG) system using LangGraph and LangChain.
It intelligently decides when to retrieve external knowledge, filters for relevant documents, generates answers, and verifies them for hallucination and usefulness before presenting a final response.

## Overview

The system models a complex state machine (using `StateGraph`) where an LLM acts as an agent to continually evaluate its own progress. By introducing self-reflection steps, the generator avoids outputting unsupported facts and will iteratively revise its answer or rewrite its search query if the generated output is not satisfactory.

## Architecture & Workflow

The LangGraph workflow consists of several specialized nodes:

1. **`decide_retrieval`**: Determines if the user's question can be answered from general knowledge or if it requires specific context retrieval.
2. **`generate_direct`**: If no retrieval is needed, the LLM generates an answer directly.
3. **`retrieve`**: Fetches similar documents from a local FAISS vector store.
4. **`is_relevant`**: Evaluates each retrieved document to ensure it accurately matches the topic of the user's question, filtering out noise.
5. **`generate_from_context`**: Generates a preliminary answer strictly based on the filtered relevant documents.
6. **`is_sup` (Is Supported)**: Checks if the generated answer is strictly grounded in the retrieved context. It categorizes the support as `fully_supported`, `partially_supported`, or `no_support`.
7. **`revise_answer`**: If the answer is not fully supported, the system falls back to quoting the context directly to eliminate hallucinations.
8. **`is_use` (Is Useful)**: Verifies if the fully supported answer _actually answers the user's original question_.
9. **`rewrite_question`**: If the answer is grounded but not useful (e.g., missed the specific detail requested), the LLM rewrites the retrieval query to fetch better documents and loops back to the retrieval phase.

## Requirements

Ensure you have a `.env` file containing your valid API keys:

```
GOOGLE_API_KEY=your_google_api_key
```

### Dependencies

You will need the following Python packages:

- `langchain`, `langchain-google-genai`, `langchain-huggingface`
- `langgraph`
- `faiss-cpu` (for local vector storage)
- `sentence-transformers` (for embeddings)
- `pypdf` (for loading PDF documents)
- `pydantic`

## Setup

1. Place your data resources (PDFs) in the `./docs/` folder (e.g., `Company_Policies.pdf`, `Company_Profile.pdf`, `Product_and_Pricing.pdf`).
2. Run the notebook `self_rag.ipynb` to ingest the PDFs, chunk them, embed them, and store them in the FAISS vector database.
3. The final cell in the notebook initializes the graph state and invokes the LangGraph application.

## Key Features

- **Hallucination Prevention**: The system enforces strict adherence to source material using the `is_sup` check.
- **Self-Correction**: If the pipeline realizes its answer failed to satisfy the user prompt, it automatically regenerates a better semantic search query (`rewrite_question`).
- **Adaptive Fallbacks**: An internal loop allows for up to `recursion_limit` attempts to fix answers before it gracefully degrades (e.g., declaring "No answer found.").
