# local RAG agent (a part of my toolkit)

This is an internal tool that is not meant to be exposed to the internet (at least not without using the authentication), because it lacks input sanitization, input classifier, output classifier, rate limiter, session control, secure headers, CSRF protection, CORS policy and other security features.

The idea was to make a smart semantic search over my Zotero library. Very scalable incremental ingestion of a directory with pdfs (of those that are not already in the database). `PyPDFLoader` reads a pdf, `RecursiveCharacterTextSplitter` with `thenlper/gte-small` tokenizer (to avoid splitting mid-word) splits the text into chunks and Chroma ingests them with the metadata and embeddings calculated by `sentence-transformers/all-mpnet-base-v2`. Then `ToolCallingAgent` use LLM to form queries to the database and run the LLM over the retrieved content. All running locally. Most of the time have to explicitly tell the model to use the tool.

> [!WARNING]
> work in progress

**needs** a GPU with 24GB of VRAM and ollama instance running in a docker container or locally

## software stack:

- smolagents
- ollama
- chromadb
- langchain
- gradio

## todo

- [x] hybrid search with Cosine distance and Maximum Marginal Relevance (MMR)
- [x] the tool returns only part of the metadata for some reason (that was because the old version of chromadb was used when ingesting the data)

## deploy

- create a model with a larger context window:

```bash
docker exec ollama sh -c "echo 'FROM qwen2.5:7b-instruct-q4_K_M\nPARAMETER num_ctx 32768' | tee qwen2.5:7b-instruct-q4_K_M-32k.cfg"
docker exec ollama ollama create qwen2.5:7b-instruct-q4_K_M-32k -f qwen2.5:7b-instruct-q4_K_M-32k.cfg
```

- update `.env` with the following:

```
TOOL_MODEL_ID=qwen2.5:7b-instruct-q4_K_M-32k
SMOLAGENT_USERNAME=
SMOLAGENT_PASSWORD=
OLLAMA_URL=http://ollama:11434/v1
```

- deploy with docker:

`mkdir ~/shared`

```bash
docker compose up -d --build --force-recreate
docker exec -it rag-smolagent python ingest_pdfs.py
docker restart rag-smolagent
```

**or locally**

- env:

```bash
conda create --name smolagent python=3.10 -y
conda activate smolagent
pip install --upgrade pip
pip install --timeout=100 --index-url https://pypi.tuna.tsinghua.edu.cn/simple --find-links https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html torch===2.2.2+cu118 torchaudio===2.2.2+cu118
pip install -r requirements.txt
```

- copy pdfs to `data/`

- run:

```bash
python ingest_pdfs.py
GRADIO_SERVER_NAME="0.0.0.0" GRADIO_SHARE="False" GRADIO_ANALYTICS_ENABLED="False" python smolagent_rag.py
```

- open `http://localhost:7860/` in a browser

# references

- https://docs.trychroma.com/docs/querying-collections/metadata-filtering
- https://github.com/AgentOps-AI/agentops/blob/main/examples/smolagents_examples/multi_smolagents_system.ipynb
- https://github.com/huggingface/smolagents/blob/main/examples/rag_using_chromadb.py
- https://github.com/coleam00/ottomator-agents/tree/main/r1-distill-rag
- https://github.com/ernanhughes/deepresearch/blob/main/deepresearch/app.py
- https://github.com/RvTechiNNovate/Vector-DB-Handbook/
- https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf
- https://kaustavmukherjee-66179.medium.com/improve-retrieval-of-documents-from-vectordb-using-maximum-marginal-relevance-mmr-for-balancing-f6ae56fb9512
- https://docs.trychroma.com/docs/collections/configure
- https://huggingface.co/blog/gradio-5-security
