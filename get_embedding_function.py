from langchain_ollama import OllamaEmbeddings

def get_embedding_function():

    embeddings = OllamaEmbeddings(model="llama3.1:8b")
    return embeddings