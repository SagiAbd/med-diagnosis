import json
import base64
from typing import List, AsyncGenerator
from sqlalchemy.orm import Session
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from app.core.config import settings
from app.models.chat import Message
from langchain.globals import set_verbose, set_debug
from app.services.vector_store import VectorStoreFactory
from app.services.embedding.embedding_factory import EmbeddingsFactory
from app.services.llm.llm_factory import LLMFactory

set_verbose(True)
set_debug(True)

async def generate_response(
    query: str,
    messages: dict,
    knowledge_base_ids: List[int],
    chat_id: int,
    db: Session
) -> AsyncGenerator[str, None]:
    try:
        # Create user message
        user_message = Message(
            content=query,
            role="user",
            chat_id=chat_id
        )
        db.add(user_message)
        db.commit()
        
        # Create bot message placeholder
        bot_message = Message(
            content="",
            role="assistant",
            chat_id=chat_id
        )
        db.add(bot_message)
        db.commit()
        
        # Use the single fixed collection for all queries
        embeddings = EmbeddingsFactory.create()
        vector_store = VectorStoreFactory.create(
            store_type=settings.VECTOR_STORE_TYPE,
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
        )

        count = vector_store._store._collection.count()
        print(f"Collection '{settings.CHROMA_COLLECTION_NAME}' count:", count)

        if count == 0:
            error_msg = "The knowledge base is empty. Please ensure corpus.json is loaded on startup."
            yield f'0:"{error_msg}"\n'
            yield 'd:{"finishReason":"stop","usage":{"promptTokens":0,"completionTokens":0}}\n'
            bot_message.content = error_msg
            db.commit()
            return

        retriever = vector_store.as_retriever()
        
        # Initialize the language model
        llm = LLMFactory.create()
        
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
            "You are given a user question, and please write clean, concise and accurate answer to the question. "
            "You will be given a set of related contexts to the question, which are numbered sequentially starting from 1. "
            "Each context has an implicit reference number based on its position in the array (first context is 1, second is 2, etc.). "
            "Please use these contexts and cite them using the format [citation:x] at the end of each sentence where applicable. "
            "Your answer must be correct, accurate and written by an expert using an unbiased and professional tone. "
            "Please limit to 1024 tokens. Do not give any information that is not related to the question, and do not repeat. "
            "Say 'information is missing on' followed by the related topic, if the given context do not provide sufficient information. "
            "If a sentence draws from multiple contexts, please list all applicable citations, like [citation:1][citation:2]. "
            "Other than code and specific names and citations, your answer must be written in the same language as the question. "
            "Be concise.\n\nContext: {context}\n\n"
            "Remember: Cite contexts by their position number (1 for first context, 2 for second, etc.) and don't blindly "
            "repeat the contexts verbatim."
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        # 修改 create_stuff_documents_chain 来自定义 context 格式
        document_prompt = PromptTemplate.from_template("\n\n- {page_content}\n\n")

        # Create QA chain
        question_answer_chain = create_stuff_documents_chain(
            llm,
            qa_prompt,
            document_variable_name="context",
            document_prompt=document_prompt
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
                # if include __LLM_RESPONSE__, only use the last part
                if "__LLM_RESPONSE__" in message["content"]:
                    message["content"] = message["content"].split("__LLM_RESPONSE__")[-1]
                chat_history.append(AIMessage(content=message["content"]))

        full_response = ""
        async for chunk in rag_chain.astream({
            "input": query,
            "chat_history": chat_history
        }):
            if "context" in chunk:
                serializable_context = []
                for context in chunk["context"]:
                    serializable_doc = {
                        "page_content": context.page_content.replace('"', '\\"'),
                        "metadata": context.metadata,
                    }
                    serializable_context.append(serializable_doc)
                
                # 先替换引号，再序列化
                escaped_context = json.dumps({
                    "context": serializable_context
                })

                # 转成 base64
                base64_context = base64.b64encode(escaped_context.encode()).decode()

                # 连接符号
                separator = "__LLM_RESPONSE__"
                
                yield f'0:"{base64_context}{separator}"\n'
                full_response += base64_context + separator

            if "answer" in chunk:
                answer_chunk = chunk["answer"]
                full_response += answer_chunk
                # Escape quotes and use json.dumps to properly handle special characters
                escaped_chunk = (answer_chunk
                    .replace('"', '\\"')
                    .replace('\n', '\\n'))
                yield f'0:"{escaped_chunk}"\n'
            
        # Update bot message content
        bot_message.content = full_response
        db.commit()
            
    except Exception as e:
        error_message = f"Error generating response: {str(e)}"
        print(error_message)
        yield '3:{text}\n'.format(text=error_message)
        
        # Update bot message with error
        if 'bot_message' in locals():
            bot_message.content = error_message
            db.commit()
    finally:
        db.close()