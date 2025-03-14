from smolagents import OpenAIServerModel, ToolCallingAgent, GradioUI, Tool
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import MergerRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers.embeddings_redundant_filter import EmbeddingsRedundantFilter
from langchain_community.document_transformers import LongContextReorder
import os
import time
from typing import Tuple, List






def secure_auth_pairs() -> List[Tuple[str, str]]:
    logins = os.getenv("SMOLAGENT_USERNAME", "").split()
    passwords = os.getenv("SMOLAGENT_PASSWORD", "").split()
    if len(logins) != len(passwords):
        raise ValueError("Number of logins and passwords must match")
    return list(zip(logins, passwords))

# class RetrieverTool(Tool): # this one is for simple cosine similarity
#     name = "retriever"
#     description = (
#         "Uses semantic search to retrieve the parts of documents that could be most relevant to answer your query."
#     )
#     inputs = {
#         "query": {
#             "type": "string",
#             "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
#         }
#     }
#     output_type = "string"

#     def __init__(self, vector_store, **kwargs):
#         super().__init__(**kwargs)
#         self.vector_store = vector_store

#     def forward(self, query: str) -> str:
#         assert isinstance(query, str), "Your search query must be a string"
#         docs = self.vector_store.similarity_search(query, k=10)
#         return "\nRetrieved documents:\n" + "".join(
#             [f"\n\n===== {doc.metadata}=====\n" + doc.page_content for i, doc in enumerate(docs)]
        
#         )

class RetrieverTool(Tool): # https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf
    name = "retriever"
    description = (
        "Uses semantic search to retrieve the parts of documents that could be most relevant to answer your query."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, vector_store, embeddings, **kwargs):
        super().__init__(**kwargs)
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.hybrid_retriever = self._setup_hybrid_retriever(vector_store, embeddings)

    def _setup_hybrid_retriever(self, vector_store, embeddings):
        # Create similarity retriever
        retriever_sim = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 10}
        )
        
        # Create MMR retriever
        retriever_mmr = vector_store.as_retriever(
            search_type="mmr", search_kwargs={"k": 10}
        )
        
        # Combine retrievers
        merger = MergerRetriever(retrievers=[retriever_sim, retriever_mmr])
        
        # Set up filtering and reordering
        filter = EmbeddingsRedundantFilter(embeddings=embeddings)
        reordering = LongContextReorder()
        pipeline = DocumentCompressorPipeline(transformers=[filter, reordering])
        
        # Create compression retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline, base_retriever=merger
        )
        
        return compression_retriever

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"
        
        # Use hybrid retrieval
        docs = self.hybrid_retriever.invoke(query)
        
        return "\nRetrieved documents:\n" + "".join(
            [f"\n\n===== Document start: {doc.metadata}=====\n" + doc.page_content + "\n===== Document end.=====\n" for doc in docs]
        )




def main():
    load_dotenv()

    OLLAMA_URL = os.getenv("OLLAMA_URL")
    tool_model_id = os.getenv("TOOL_MODEL_ID")
    auth_pairs = secure_auth_pairs()

    # Initialize vector store and embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cuda'}
        )

    db_dir = os.path.join("chroma_db")
    while not os.path.exists(db_dir):
        time.sleep(60)
    vectordb = Chroma(collection_name="my_pdfs", 
                    persist_directory=db_dir, 
                    embedding_function=embeddings,
                    collection_metadata={"hnsw:space": "cosine", "hnsw:construction_ef": 200},
                    create_collection_if_not_exists=False)
    retriever_tool = RetrieverTool(vectordb, embeddings)



    model = OpenAIServerModel(
                model_id=tool_model_id,
                api_base=OLLAMA_URL,
                api_key="ollama",
                max_tokens=32768,
                temperature=0.0,
            )


    agent = ToolCallingAgent(
        tools=[retriever_tool],
        model=model,
        add_base_tools=False,
        max_steps=1,
    )

    GradioUI(agent).launch(auth=auth_pairs if len(auth_pairs) > 0 else None, pwa=True)

if __name__ == "__main__":
    main()  