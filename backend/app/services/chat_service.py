from typing import List, AsyncGenerator
from sqlalchemy.orm import Session
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.models.knowledge import KnowledgeBase, Document
from langchain_core.messages import HumanMessage, AIMessage
from app.core.config import settings

async def generate_response(
    query: str,
    messages: dict,
    knowledge_base_ids: List[int], 
    db: Session
) -> AsyncGenerator[str, None]:
    try:
        # Get knowledge bases and their documents
        knowledge_bases = (
            db.query(KnowledgeBase)
            .filter(KnowledgeBase.id.in_(knowledge_base_ids))
            .all()
        )
        
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_API_BASE
        )
        
        # Create a vector store for each knowledge base
        vector_stores = []
        for kb in knowledge_bases:
          documents = db.query(Document).filter(Document.knowledge_base_id == kb.id).all()
          if documents:
              vector_store = Chroma(
                  collection_name=f"kb_{kb.id}",
                  embedding_function=embeddings,
                  persist_directory="./chroma_db"
              )
              # 添加调试打印
              print(f"Collection {f'kb_{kb.id}'} count:", vector_store._collection.count())
              vector_stores.append(vector_store)
        
        if not vector_stores:
            yield "I don't have any knowledge base to help answer your question."
            return
        
        # Use first vector store for now： 
        # https://python.langchain.com/api_reference/langchain/retrievers/langchain.retrievers.multi_vector.MultiVectorRetriever.html
        retriever = vector_stores[0].as_retriever()
        
        # Initialize the language model
        llm = ChatOpenAI(
            temperature=0,
            streaming=True,
            model="gpt-4o",
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_API_BASE
        )

        # Create contextualize question prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, just "
            "reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        # Create history aware retriever
        history_aware_retriever = create_history_aware_retriever(
            llm, 
            retriever,
            contextualize_q_prompt
        )

        # Create QA prompt
        qa_system_prompt = (
            "You are an assistant for question-answering tasks. Use "
            "the following pieces of retrieved context to answer the "
            "question. If you don't know the answer, just say that you "
            "don't know. Use three sentences maximum and keep the answer "
            "concise.\n\nContext: {context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        # Create QA chain
        question_answer_chain = create_stuff_documents_chain(
            llm,
            qa_prompt
        )

        # Create retrieval chain
        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain,
        )

        # Generate response
        chat_history = []
        for message in messages["messages"]:
            if message["role"] == "user":
                chat_history.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                chat_history.append(AIMessage(content=message["content"]))

        async for chunk in rag_chain.astream({
            "input": query,
            "chat_history": chat_history
        }):
            if "answer" in chunk:
                yield chunk["answer"]
            
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        yield f"Error generating response: {str(e)}"