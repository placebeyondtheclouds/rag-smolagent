from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
import glob
from tqdm import tqdm
from transformers import AutoTokenizer
import torch

load_dotenv()

def ingest_pdfs_incrementally(data_dir: str, persist_directory: str):

    pdf_file_paths = glob.glob(os.path.join(data_dir, "**", "*.pdf"), recursive=True)



    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2", #Max Sequence Length: 384, Dimensions: 768
        model_kwargs={'device': 'cuda'}
    )

    print("Initializing vector store...")
    vectordb = Chroma(
        collection_name="my_pdfs",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )


    with torch.device("cuda"):
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained("thenlper/gte-small"),
            chunk_size=200,
            chunk_overlap=20,
            add_start_index=True,
            strip_whitespace=True,
            separators=["\n\n", "\n", ".", " ", '。', '！', '!', '？', '?'],
            )

    for pdf_file in tqdm(pdf_file_paths, desc="Processing PDFs"):
        if len(vectordb.get(where={"source": {"$eq": pdf_file}})['ids'])>0:
            print(f"Skipping {pdf_file} as it has already been processed")
            continue
        try:
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()

            chunks = text_splitter.split_documents(docs)
            if chunks:
                print(f"Adding {len(chunks)} chunks from {pdf_file}")
                vectordb.add_documents(documents=chunks)

        except Exception as e:
            print(f"Error loading PDF file: {pdf_file}, error: {str(e)}")
            continue


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    db_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
    
    print("Incrementally ingesting PDFs into the vector store...")
    ingest_pdfs_incrementally(data_dir, db_dir)
    print(f"Vector store created and persisted at {db_dir}")

if __name__ == "__main__":
    main()