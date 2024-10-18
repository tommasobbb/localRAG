import argparse
import os
import shutil
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
import subprocess
from query_functions import query_rag
import requests


OLLAMA_SERVER_COMMAND = ["ollama", "serve"]  # Example command to start the Ollama server
CHROMA_PATH = "chroma"
DATA_PATH = "data"
DATA_PATH_1 = "/Users/tommaso/Downloads"


def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Start the Ollama server
    start_ollama_server()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

    # Enter chatbot mode
    start_chatbot()


def start_ollama_server():
    """Starts the Ollama local server if it's not already running."""
    try:
        # Check if the server is already running
        response = requests.get('http://localhost:11434/models')
        if response.status_code == 200:
            print("✅ Ollama server is already running.")
            return
    except requests.ConnectionError:
        print("🚀 Starting Ollama server...")
        try:
            subprocess.Popen(OLLAMA_SERVER_COMMAND, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("✅ Ollama server started successfully.")
        except Exception as e:
            print(f"❌ Failed to start Ollama server: {str(e)}")


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def start_chatbot():
    """Starts a loop asking for queries and responding using query_rag."""
    print("\n💬 Enter 'exit' to stop the chatbot.")
    while True:
        query = input("\n📝 Ask a question: ")
        if query.lower() == "exit":
            print("👋 Exiting chatbot.")
            break
        response = query_rag(query)
        print(f"🤖 {response}\n")

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()