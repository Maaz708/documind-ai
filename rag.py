from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from ingest import get_embeddings
import config
import prompts

def parse_chat_history(history_list: list) -> list:
    """
    Parses a list of message dicts (e.g. [{"role": "user", "content": "..."}])
    into LangChain message instances (HumanMessage, AIMessage).
    """
    messages = []
    for msg in history_list:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages

def format_docs(docs: list) -> str:
    """
    Formats list of documents into a single block of text with page number references.
    """
    formatted = []
    for doc in docs:
        page_num = doc.metadata.get("page", 0) + 1  # Metadata page numbers are 0-indexed
        formatted.append(f"[Page {page_num}]: {doc.page_content}")
    return "\n\n".join(formatted)

def query_rag(doc_id: str, question: str, chat_history_list: list) -> dict:
    """
    Executes the complete conversational RAG query flow:
    1. Parse history.
    2. Condense follow-up questions to standalone search queries.
    3. Retrieve relevant context from Pinecone for the specified document ID.
    4. Generate response with context, history, and source mappings.
    """
    config.validate_config()
    
    # Initialize embeddings and Pinecone vector store
    embeddings = get_embeddings()
    vector_store = PineconeVectorStore(
        index_name=config.PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=doc_id
    )
    
    # Retrieve top 4 most relevant chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # Initialize LLM
    llm = ChatGroq(
        model=config.GROQ_MODEL,
        groq_api_key=config.GROQ_API_KEY,
        temperature=0.2
    )
    
    # Parse incoming chat history
    chat_history = parse_chat_history(chat_history_list)
    
    # Reformulate question if chat history exists
    standalone_question = question
    if chat_history:
        condense_chain = prompts.CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()
        try:
            standalone_question = condense_chain.invoke({
                "chat_history": chat_history,
                "question": question
            })
            print(f"[RAG] Reformulated: '{question}' -> Standalone: '{standalone_question}'")
        except Exception as e:
            print(f"[RAG] Failed to reformulate, using original question. Error: {str(e)}")
            
    # Retrieve documents
    retrieved_docs = retriever.invoke(standalone_question)
    context_str = format_docs(retrieved_docs)
    
    # Generate final QA response
    qa_chain = prompts.QA_PROMPT | llm | StrOutputParser()
    answer = qa_chain.invoke({
        "context": context_str,
        "chat_history": chat_history,
        "question": question
    })
    
    # Extract source metadata mapping for the UI
    sources = []
    for doc in retrieved_docs:
        page_num = doc.metadata.get("page", 0) + 1
        # Extract preview snippet of the source block
        snippet = doc.page_content[:200]
        if len(doc.page_content) > 200:
            snippet += "..."
        sources.append({
            "page": page_num,
            "content": snippet
        })
        
    return {
        "answer": answer,
        "sources": sources
    }
