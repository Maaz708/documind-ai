import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

import config

def get_embeddings():
    """
    Returns the appropriate LangChain embedding instance based on config.
    """
    config.validate_config()
    
    if config.EMBEDDING_PROVIDER == "huggingface-api":
        from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
        return HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            task="feature-extraction",
            huggingfacehub_api_token=config.HF_TOKEN
        )
    elif config.EMBEDDING_PROVIDER == "huggingface-local":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    elif config.EMBEDDING_PROVIDER == "google-api":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=config.GOOGLE_API_KEY
        )
    else:
        raise ValueError(f"Unknown embedding provider: {config.EMBEDDING_PROVIDER}")

def get_embedding_dimension() -> int:
    """
    Returns the dimension of the embedding vectors produced by the selected model.
    """
    if config.EMBEDDING_PROVIDER in ["huggingface-api", "huggingface-local"]:
        return 384
    elif config.EMBEDDING_PROVIDER == "google-api":
        return 768
    else:
        raise ValueError(f"Unknown embedding provider: {config.EMBEDDING_PROVIDER}")

def setup_pinecone_index():
    """
    Connects to Pinecone and initializes the index if it does not exist.
    """
    config.validate_config()
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    
    index_name = config.PINECONE_INDEX_NAME
    dimension = get_embedding_dimension()
    
    # Check if index exists
    if not pc.has_index(index_name):
        print(f"Creating Pinecone index '{index_name}' with dimension {dimension}...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"  # AWS us-east-1 is typical for Pinecone free tier serverless
            )
        )
        # Wait for index to be ready
        while not pc.describe_index(index_name).status.ready:
            print("Waiting for Pinecone index to be ready...")
            time.sleep(2)
        print("Pinecone index is ready.")
    else:
        # If index exists, verify dimension matches
        desc = pc.describe_index(index_name)
        if desc.dimension != dimension:
            print(
                f"WARNING: Pinecone index '{index_name}' has dimension {desc.dimension}, "
                f"but embedding provider '{config.EMBEDDING_PROVIDER}' produces dimension {dimension}. "
                f"This may lead to query errors."
            )
            
    return pc.Index(index_name)

def ingest_pdf(file_path: str, doc_id: str):
    """
    Loads a PDF file, splits it into chunks, and uploads the vectors to Pinecone
    using doc_id as the isolated index namespace.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    print(f"Loading PDF: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    print("Splitting document text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(documents)
    
    # Inject metadata to identify source details
    for chunk in chunks:
        chunk.metadata["doc_id"] = doc_id
        
    embeddings = get_embeddings()
    setup_pinecone_index()
    
    print(f"Uploading {len(chunks)} text chunks to Pinecone under namespace '{doc_id}'...")
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=config.PINECONE_INDEX_NAME,
        namespace=doc_id
    )
    print("Vector storage ingestion successful.")
