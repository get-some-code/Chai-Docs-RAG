
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import google.generativeai as genai
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Vector Embeddings
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GEMINI_API_KEY
)

# Load vector store
vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="chaicode-docs",
    embedding=embedding_model
)

# Initialize Gemini model
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Maintain history of chat
messages = []

# Chat session
chat = model.start_chat(history=messages)

# Loop
print("Ask any questions from Chai Docs:\n\n")
while True:
    query = input("> ")
    if query.lower().strip() in ['bye','see ya','brb','exit']:
        break
    elif query.lower().strip() in ["hi", "hii", "hello", "hey","how are you"]:
        print("🤖: Namaste! Ask me something from Chai aur Docs 📚")
        continue

    # Vector similarity search in DB
    search_results = vector_db.similarity_search(query=query)
    # print(search_results)

    # Create context from search results
    context = "\n\n\n".join([
        f"Page Title: {result.metadata['title']}\nContents: {result.page_content}\nSource: {result.metadata['source']}\n"
        for result in search_results
    ])

    # print(context)

    # # System prompt based on context
    SYSTEM_PROMPT = f"""
    You are a helpful and knowledgeable AI Assistant trained to answer user queries using only the provided documentation context retrieved from Chai aur Code Docs.

    Your responsibilities:
    - Answer **only** based on the given context. If the answer is not in the context, politely say so.
    - If the documentation contains **examples**, include them in your answer to improve clarity.
    - Respond in a clear, structured, and beginner-friendly tone.
    - Mention the **source link** so the user can refer to the original doc.
    - Avoid hallucinating or using external information not found in the context.

    *** Avoid responses like:

    User: How are you?  
    Assistant: I'm sorry, but the provided text does not contain information about how I am. The document only shows page titles and the phrase "Welcome | Chai aur Docs" repeated multiple times.

    *** Instead, respond like:

    User: Hi, how are you?  
    Assistant: Hello! Please ask relevant questions from the Chai aur Code documentation so I can help you better.

    ---

    Here is the documentation context:
    {context}

"""

    # Add system instruction and user query to history
    chat.send_message(f"{SYSTEM_PROMPT}\n\nUser Query: {query}")

    # Get model response
    response = chat.last.text

    # Show response
    print("\n🤖:", response)
