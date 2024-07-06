from langchain.embeddings import VertexAIEmbeddings

class EmbeddingClient:
    def __init__(self, model_name, project, location):
        self.client = VertexAIEmbeddings(
            model_name=model_name,
            project=project,
            location=location
        )
        
    def embed_query(self, query):
        try:
            vectors = self.client.embed_query(query)
            return vectors
        except Exception as e:
            print(f"An error occurred while embedding the query: {e}")
            return None
        
    def embed_documents(self, documents):
        try:
            return self.client.embed_documents(documents)
        except AttributeError:
            print("Method embed_documents not defined for the client.")
            return None
        except Exception as e:
            print(f"An error occurred while embedding the documents: {e}")
            return None

if __name__ == "__main__":
    model_name = "textembedding-gecko@003"
    project = "gemini-quizzify-428422"
    location = "us-central1"
    
    embedding_client = EmbeddingClient(model_name, project, location)
    vectors = embedding_client.embed_query("Hello World!")
    if vectors:
        print(vectors)
        print("Successfully used the embedding client!")
    else:
        print("Failed to embed the query.")
