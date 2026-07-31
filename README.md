# Q&A (RAG Chatbot)

A Retrieval-Augmented Generation (RAG) chatbot that can answer questions about any PDF you upload, built with Ollama and pypdf, ChromaDB, and sentence-transformers libraries, with a simple Streamlit UI for accessibility.

## Motivation

While searching for an explanation of fantasy author Brandon Sanderson's fictional universe, the Cosmere, I found existing explanations written by fans unhelpful. I eventually found a large essay written by Sanderson himself. This prompted me to build a chatbot that could answer my specific questions about the Cosmere directly from his own writing, rather than relying on secondhand summaries. The same system works with any PDF the user uploads.

## How it works

Like any RAG system, this project:
1. Splits the PDF's text into overlapping chunks
2. Converts each chunk into a vector embedding
3. Stores the embeddings in a vector database
4. Finds the most relevant chunks for a given question using cosine similarity
5. Passes that context to Ollama, which generates a coherent, grounded answer

## Key features

- **Cosine similarity search** — retrieves chunks that are semantically relevant to the question, not just keyword matches
- **Overlapping chunks** — each chunk is 500 characters long with a 50-character overlap, so sentences aren't cut awkwardly between chunks; the overlap also means related content appears in multiple chunks, helping the model place them closer together in vector space
- **Hallucination resistance** — the model is explicitly instructed to say when the provided context doesn't contain the answer, rather than guessing
- **Standalone testing** — `rag.py` includes a commented-out block for testing the pipeline directly in the terminal, without needing the Streamlit UI
- **Reproducibility** — all dependencies are pinned in `requirements.txt` for easy setup on any machine

## Example queries

These test questions demonstrate the model correctly answering from context, admitting uncertainty, and refusing to answer questions unrelated to the document:

```
> who is the author of the cosmere?
Brandon Sanderson.

> How many trilogies have been written about the cosmere? name them.
The context doesn't provide information on the number of trilogies or specific 
trilogies written about the Cosmere. It only provides a general introduction to 
the Cosmere as an interconnected universe with epic fantasy stories, including 
the Mistborn series.

> give a one-sentence description about the mistborn series
The context does not provide a specific description for the Mistborn series, 
only that it is part of the Cosmere and was written with its connections in mind.

> what is the plot of harry potter?
The context does not mention Harry Potter, nor does it provide any information 
about its plot. The discussion appears to be centered around Brandon Sanderson's 
Cosmere and his approach to world-building in his series.
```

Note the model correctly declines to answer when the source document doesn't contain the information — this is the hallucination-resistance feature in action, not a limitation.

## Tech stack

- **Python**
- **Streamlit** — web UI
- **ChromaDB** — local vector database
- **sentence-transformers** — text embeddings
- **Ollama (Llama 3.2)** — local LLM for answer generation, no API key or cost required
- **pypdf** — PDF text extraction

## Running it locally

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/document-qa-rag.git
cd document-qa-rag
```

**2. Install Ollama and pull the model**

Download from [ollama.com](https://ollama.com/download), then:
```bash
ollama pull llama3.2
```

**3. Install Python dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
streamlit run app.py
```

A local website will open in your browser — upload a PDF and start asking questions.

## Project structure

```
document-qa-rag/
├── rag.py           # core RAG pipeline: chunking, embedding, retrieval, answering
├── app.py           # Streamlit UI
├── requirements.txt
└── README.md
```

## Limitations & future improvements

- Currently supports one document at a time; multi-document support with source citations would be a natural next step
- Chunk size (500 characters) was chosen as a reasonable default and hasn't been systematically tuned against answer quality


