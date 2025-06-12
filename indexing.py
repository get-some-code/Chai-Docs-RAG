from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from pathlib import Path
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
import re

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Function to remove surrogate unicode characters that cause encoding errors
def remove_surrogates(text: str) -> str:
    return re.sub(r'[\ud800-\udfff]', '', text)

# Loading PDF FILES
urls = [
    "https://docs.chaicode.com/youtube/getting-started/",
    "https://docs.chaicode.com/youtube/chai-aur-html/welcome/",
    "https://docs.chaicode.com/youtube/chai-aur-html/introduction/",
    "https://docs.chaicode.com/youtube/chai-aur-html/emmit-crash-course/",
    "https://docs.chaicode.com/youtube/chai-aur-html/html-tags/",
    "https://docs.chaicode.com/youtube/chai-aur-git/welcome/",
    "https://docs.chaicode.com/youtube/chai-aur-git/introduction/",
    "https://docs.chaicode.com/youtube/chai-aur-git/terminology/",
    "https://docs.chaicode.com/youtube/chai-aur-git/behind-the-scenes/",
    "https://docs.chaicode.com/youtube/chai-aur-git/branches/",
    "https://docs.chaicode.com/youtube/chai-aur-git/diff-stash-tags/",
    "https://docs.chaicode.com/youtube/chai-aur-git/managing-history/",
    "https://docs.chaicode.com/youtube/chai-aur-git/github/",
    "https://docs.chaicode.com/youtube/chai-aur-c/welcome/",
    "https://docs.chaicode.com/youtube/chai-aur-c/introduction/",
    "https://docs.chaicode.com/youtube/chai-aur-c/hello-world/",
    "https://docs.chaicode.com/youtube/chai-aur-c/variables-and-constants/",
    "https://docs.chaicode.com/youtube/chai-aur-c/data-types/",
    "https://docs.chaicode.com/youtube/chai-aur-c/operators/",
    "https://docs.chaicode.com/youtube/chai-aur-c/control-flow/",
    "https://docs.chaicode.com/youtube/chai-aur-c/loops/",
    "https://docs.chaicode.com/youtube/chai-aur-c/functions/",
    "https://docs.chaicode.com/youtube/chai-aur-django/welcome/",
    "https://docs.chaicode.com/youtube/chai-aur-django/getting-started/",
    "https://docs.chaicode.com/youtube/chai-aur-django/jinja-templates/",
    "https://docs.chaicode.com/youtube/chai-aur-django/tailwind/",
    "https://docs.chaicode.com/youtube/chai-aur-django/models/",
    "https://docs.chaicode.com/youtube/chai-aur-django/relationships-and-forms/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/welcome/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/introduction/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/postgres/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/normalization/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/database-design-exercise/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/joins-and-keys/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/joins-exercise/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/welcome/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/setup-vpc/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/setup-nginx/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/nginx-rate-limiting/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/nginx-ssl-setup/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/node-nginx-vps/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/postgresql-docker/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/postgresql-vps/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/node-logger/"
]


loader = WebBaseLoader(web_paths=urls)
docs = loader.load()

# print(docs[25])

# Clean the page content to remove surrogates before splitting
for doc in docs:
    doc.page_content = remove_surrogates(doc.page_content)

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=500
)
splitted_docs = text_splitter.split_documents(documents=docs)

# Also clean each chunk's content just in case
for chunk in splitted_docs:
    chunk.page_content = remove_surrogates(chunk.page_content)

# Vector Embeddings
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GEMINI_API_KEY
)

vector_store = QdrantVectorStore.from_documents(
    documents=splitted_docs,
    url="http://localhost:6333",
    collection_name="chaicode-docs",
    embedding=embedding_model,
    force_recreate=True  # Recreate collection to avoid dimension mismatch errors
)

print("Indexing of Documents Done...")
