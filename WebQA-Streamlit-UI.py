
import streamlit as st
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

import os
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

@st.cache_resource
def load_llm_embeddings():
    llm = OpenAI()
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    return llm, embeddings

llm, embeddings = load_llm_embeddings()

@st.cache_resource
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
    chat_history.append((question, response))
    return response, chat_history


st.title("Webpage Question Answering System 🌐")

url = st.text_input("Enter the URL of the website:")

def clear_chat_history():
    st.session_state.messages = []

if url:
    if st.session_state.get("uploaded_url_name") != url:
        clear_chat_history()
        st.session_state.uploaded_url_name = url
    chat_history = []
    text_chunks = load_data(url)

    if not text_chunks:
        st.warning("The website content is not available or does not contain relevant information.")
    else:
        ensemble_retriever = ensemble_retrievers(text_chunks, embeddings)
        llm_chain = qa_pipeline(ensemble_retriever)

        if prompt := st.chat_input("Your question"):
            st.session_state.messages.append({"role": "user", "content": prompt})

        if "messages" in st.session_state and st.session_state.messages:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

        if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message(name="assistant"):
                with st.spinner("Thinking..."):
                    response, chat_history = interact_with_chatbot(llm_chain, prompt, chat_history)
                    st.write(response)
                    message = {"role": "assistant", "content": response}
                    st.session_state.messages.append(message)

if st.button("Clear Chat History"):
    clear_chat_history()
