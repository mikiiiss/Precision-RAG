# import openai
# import os

# from dotenv import load_dotenv


# load_dotenv()
# openai.api_key = os.getenv('OPENAI_API_KEY')

# response = openai.Embedding.create(
#     input="is",
#     model="text-embedding-ada-002"
# )

# embedding = response['data'][0]['embedding']
# print(embedding)
import os
import openai
import fitz  # PyMuPDF
import json
import csv
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Function to read the PDF and extract text
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Function to chunk text
def chunk_text(text, chunk_size=50):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Function to vectorize text chunks
def vectorize_chunks(chunks, model="text-embedding-ada-002"):
    embeddings = []
    for chunk in chunks:
        response = openai.Embedding.create(
            input=chunk,
            model=model
        )
        embedding = response['data'][0]['embedding']
        embeddings.append(embedding)
    return embeddings
# Function to save embeddings to a CSV file including the text chunks
def save_embeddings_to_csv(embeddings, chunks, file_path):
    with open(file_path, 'w', newline='') as csvfile:
        fieldnames = ['text', 'embedding']  # Include 'text' in the column names
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()  # Write the header row
        
        # Ensure chunks and embeddings lists have the same length
        assert len(chunks) == len(embeddings), "Chunks and embeddings lists must have the same length."
        
        for chunk, embedding in zip(chunks, embeddings):
            # Convert the embedding list to a string representation (JSON format)
            embedding_str = json.dumps(embedding)
            writer.writerow({'text': chunk, 'embedding': embedding_str})  # Write both text and embedding

# Update process_pdf to pass chunks to save_embeddings_to_csv
def process_pdf(pdf_path, output_path):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    embeddings = vectorize_chunks(chunks)
    save_embeddings_to_csv(embeddings, chunks, output_path)  # Pass chunks here
    print(f"Embeddings with corresponding texts saved to {output_path}")

    # Return the chunks for later use (if needed)
    return chunks   

    # Example usage
pdf_path = '10 Academy Cohort B - Weekly Challenge_ Week - 7 (1).pdf'
output_path = 'new_embeddings.csv'
process_pdf(pdf_path, output_path)