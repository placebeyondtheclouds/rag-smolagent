from smolagents import OpenAIServerModel, ToolCallingAgent, GradioUI, Tool, LiteLLMModel
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os


load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL")
tool_model_id = os.getenv("TOOL_MODEL_ID")


class RetrieverTool(Tool):
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

    def __init__(self, vector_store, **kwargs):
        super().__init__(**kwargs)
        self.vector_store = vector_store

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"
        docs = self.vector_store.similarity_search(query, k=10)
        return "\nRetrieved documents:\n" + "".join(
            # [f"\n\n===== Title {doc.metadata['title']}, Author {doc.metadata['author']}\n===== Source {doc.metadata['source']} =====\n" + doc.page_content for i, doc in enumerate(docs)]
            [f"\n\n===== {doc.metadata}=====\n" + doc.page_content for i, doc in enumerate(docs)]
        
        )




# Initialize vector store and embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cuda'}
    )

db_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
vectordb = Chroma(collection_name="my_pdfs", persist_directory=db_dir, embedding_function=embeddings)
retriever_tool = RetrieverTool(vectordb)



model = OpenAIServerModel(
            model_id=tool_model_id,
            api_base=OLLAMA_URL,
            api_key="ollama",
            max_tokens=32768,
        )


agent = ToolCallingAgent(
    tools=[retriever_tool],
    model=model,
    add_base_tools=False,
    max_steps=1,

)


def main():
    GradioUI(agent).launch(auth=(os.getenv("SMOLAGENT_USERNAME"), os.getenv("SMOLAGENT_PASSWORD")))

if __name__ == "__main__":
    main()  