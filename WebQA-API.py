from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from pathlib import Path
import uvicorn
from fastapi import FastAPI
import os
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

os.environ["OPENAI_API_KEY"] = "YOUR-OPENAI-API-KEY"

template = """
You are an AI assistant designed to provide answers based exclusively on the context provided, without relying on any prior or base knowledge or external information.The user will give a url which will be your context. It is very important that you answer solely based on that urls content.

Your task is to use only the information given in the context to answer the user's question. It is crucial that you avoid generating any information that is not explicitly stated in the provided context, even if you possess relevant knowledge from your base training.

If the context includes relevant information from the chat history, you may consider it to better understand the question and provide a more accurate answer. However, only use the chat history if it directly pertains to the current context and question.

In case the provided website context does not contain sufficient information to answer the question, if the question cannot be satisfactorily answered using only the context, or if the question is not directly related to the information given in the context, simply respond with "The provided context does not contain enough information to answer this question."

It is essential that you refrain from providing any additional information or answers based on your own knowledge, as the goal is to rely solely on the given context.

Context: {context}
Chat History: {chat_history}
Question: {question}
Answer:
"""

prompt = PromptTemplate(template=template, input_variables=["context", "chat_history", "question"])

app = FastAPI()

@app.on_event("startup")
def load_llm_embeddings():
    llm = OpenAI()
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    return llm, embeddings

llm, embeddings = load_llm_embeddings()

def load_data(url):
    urls = [url]
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(data)
    return text_chunks

def ensemble_retrievers(text_chunks, embeddings):
    faiss_vectorstore = FAISS.from_documents(text_chunks, embeddings)
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 5})
    bm25_retriever = BM25Retriever.from_documents(text_chunks)
    bm25_retriever.k = 5
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.6, 0.4])
    return ensemble_retriever

def qa_pipeline(ensemble_retriever):
    memory = ConversationBufferWindowMemory(memory_key="chat_history", k=10, return_messages=True)
    QA = ConversationalRetrievalChain.from_llm(llm, retriever=ensemble_retriever, condense_question_prompt=prompt, chain_type="stuff", memory=memory)
    return QA

def interact_with_chatbot(QA, question, chat_history):
    result = QA({"question": question, "chat_history": chat_history})
    response = result["answer"]
    return response, chat_history


class request_body(BaseModel):
    url: str
    question:str


@app.post('/url')
def AI_Message_url(data:request_body):
    data=data.dict()
    user_prompt=data['question']
    chat_history = []
    text_chunks = load_data(data['url'])
    ensemble_retriever = ensemble_retrievers(text_chunks, embeddings)
    llm_chain=qa_pipeline(ensemble_retriever)
    response, chat_history = interact_with_chatbot(llm_chain, user_prompt, chat_history)
    return {'Output':response}

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)