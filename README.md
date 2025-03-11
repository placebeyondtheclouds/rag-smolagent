# local RAG agent (a part of my toolkit)

The idea was to make a smart semantic search over my Zotero library.

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

- [] the tool returns only part of the metadata for some reason

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
-
