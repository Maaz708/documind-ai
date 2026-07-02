from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# System prompt for RAG QA
RAG_SYSTEM_PROMPT = (
    "You are DocuMind AI, a precise and professional AI document assistant. "
    "Use the following retrieved context from the uploaded PDF to answer the question. "
    "If the answer cannot be determined from the context, state clearly: "
    "\"I couldn't find the answer in the uploaded document.\"\n\n"
    "Do not extrapolate, assume, or use external facts not contained in the context unless directly relevant to clarifying terminology. "
    "When referring to details from the context, try to summarize clearly.\n\n"
    "Retrieved Context:\n"
    "---------------------\n"
    "{context}\n"
    "---------------------"
)

# Chat prompt template for QA
QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

# System prompt to generate standalone questions from chat history
CONDENSE_SYSTEM_PROMPT = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

# Chat prompt template for condensing follow-up questions
CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CONDENSE_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
