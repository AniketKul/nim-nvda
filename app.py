import streamlit as st
import os
import time
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader, WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

# load Nvidia API Key
os.environ['NVIDIA_API_KEY'] = os.getenv('NVIDIA_API_KEY')

# Nvidia NIM inferencing
llm = ChatNVIDIA(model='meta/llama-3.1-405b-instruct')

def vector_embedding():
    if 'vectors' not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        # data ingestion
        st.session_state.loader = PyPDFDirectoryLoader('./us_census') 
        # document loading
        st.session_state.docs = st.session_state.loader.load()
        # chunk creation - works recursively
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        # splitting - taking top 30 records
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])

        print('Hi there!')

        # vector embeddings i.e. our FAISS vector database
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, 
                                                        st.session_state.embeddings)


# main prompt logic
st.title('Building RAG Document Q&A With Nvidia NIM And Langchain')

prompt = ChatPromptTemplate.from_template(
    '''
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions: {input}
    '''
)

prompt1 = st.text_input('Enter your question from documents')

if st.button('Start Document Embedding'):
    vector_embedding()
    st.write('FAISS VectorDB is ready using NVIDIAEmbeddings!')

if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start_time = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    print('Response Time: ', time.process_time() - start_time)
    st.write(response['answer'])

    # with streamlit expander
    with st.expander('View Document Similarity Search'):
        # find the relevant chunks
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('-------')