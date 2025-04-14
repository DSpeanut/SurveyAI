import chromadb

# Initialize client
client = chromadb.Client()

# Create a collection
collection = client.create_collection("survey_doc")

# Add documents
collection.add(
    documents=["This is a document", "This is another document"],
    ids=["doc1", "doc2"]
)

# Search
results = collection.query(
    query_texts=["Find similar documents"],
    n_results=2
)