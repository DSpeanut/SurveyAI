import fitz
import mypdf
import os
from openai import OpenAI
import numpy as np
# Define the path to the PDF file
pdf_path = "data/AI_Information.pdf"

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.

    Args:
    pdf_path (str): Path to the PDF file.

    Returns:
    str: Extracted text from the PDF.
    """
    # Open the PDF file
    mypdf = fitz.open(pdf_path)
    all_text = ""  # Initialize an empty string to store the extracted text
    
    # Iterate through each page in the PDF
    for page in mypdf:
        # Extract text from the current page and add spacing
        all_text += page.get_text("text") + " "

    # Return the extracted text, stripped of leading/trailing whitespace
    return all_text.strip()


def chunk_text(text, n, overlap):
    """
    Splits text into overlapping chunks.

    Args:
    text (str): The text to be chunked.
    n (int): Number of characters per chunk.
    overlap (int): Overlapping characters between chunks.

    Returns:
    List[str]: A list of text chunks.
    """
    chunks = []  # Initialize an empty list to store the chunks
    for i in range(0, len(text), n - overlap):
        # Append a chunk of text from the current index to the index + chunk size
        chunks.append(text[i:i + n])
    
    return chunks  # Return the list of text chunks


def create_embeddings(openai_client, texts):
    """
    Generates embeddings for a list of texts.

    Args:
    texts (List[str]): List of input texts.
    model (str): Embedding model.

    Returns:
    List[np.ndarray]: List of numerical embeddings.
    """
    # Create embeddings using the specified model
    model="text-embedding-3-large"
    if isinstance(texts, list):
        response = openai_client.embeddings.create(model=model, input=texts)
        return [np.array(embedding.embedding) for embedding in response.data]
    elif isinstance(texts, str):
        response = openai_client.embeddings.create(model=model, input=texts)
        return [np.array(response.data[0].embedding)]

def cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity between two vectors.

    Args:
    vec1 (np.ndarray): First vector.
    vec2 (np.ndarray): Second vector.

    Returns:
    float: Cosine similarity score.
    """

    # Compute the dot product of the two vectors
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def retrieve_relevant_chunks(openai_client,query,text_chunks, chunk_embeddings, k):
    """
    Retrieves the top-k most relevant text chunks.
    
    Args:
    query (str): User query.
    text_chunks (List[str]): List of text chunks.
    chunk_embeddings (List[np.ndarray]): Embeddings of text chunks.
    k (int): Number of top chunks to return.
    
    Returns:
    List[str]: Most relevant text chunks.
    """
    # Generate an embedding for the query - pass query as a list and get first item
    query_embedding = create_embeddings(openai_client, query)[0]
    
    # Calculate cosine similarity between the query embedding and each chunk embedding
    similarities = [cosine_similarity(query_embedding, emb) for emb in chunk_embeddings]
    
    # Get the indices of the top-k most similar chunks
    top_indices = np.argsort(similarities)[-k:][::-1]
    
    # Return the top-k most relevant text chunks
    return [text_chunks[i] for i in top_indices]