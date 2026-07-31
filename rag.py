from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import ollama

def load_pdf(path):
    reader = PdfReader(path)

    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text   

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap 
    return chunks

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_chunks(chunks):
    return model.encode(chunks).tolist()


client = chromadb.Client()
collection = client.create_collection("my_docs")
def store_chunks(chunks, embeddings):
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[str(i) for i in range(len(chunks))]
    )


def retrieve(question, k=3):
    q_embeddings = model.encode([question]).tolist()
    results = collection.query(query_embeddings=q_embeddings, n_results=k)
    return results['documents'][0]





def answer_question(question):
    context_chunks = retrieve(question)
    context = "\n\n".join(context_chunks)


    prompt = f"""Answer the question using only the context below. 
    if the answer isn't in the context, say so.

    Context:
    {context}

    Question: {question}"""

    response = ollama.chat(model='llama3.2', messages=[
        {"role": "user", "content": prompt}
    ])
       

    return response['message']['content']



def build_index(path):
    text = load_pdf(path)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    store_chunks(chunks, embeddings)

# if __name__ == "__main__":
#     build_index("C:/Users/Zoella/Desktop/project/What-is-the-Cosmere.pdf")
#     while True:
#         q = input("Ask for something (or 'quit'):")
#         if q == "quit":
#             break
#         print(answer_question(q))



