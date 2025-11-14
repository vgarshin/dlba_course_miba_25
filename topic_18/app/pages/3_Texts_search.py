#!/usr/bin/env python
# coding: utf-8

import os
import io
import json
import datetime
import streamlit as st
import urllib.request
from pathlib import Path

from langchain.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.yandex import YandexGPTEmbeddings
from langchain_community.chat_models import ChatYandexGPT
from langchain_community.llms import YandexGPT
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS


st.set_page_config(
    page_title="AI search with chat",
    page_icon="💬"
)
st.sidebar.header('Chat-bot with LLM')
st.header('AI-bot for RAG based search', divider='rainbow')

st.markdown(
    """
    To get started, you need to enter a set of parameters for the chatbot 
    and then define the subject area with a query on the topic of interest. 
    After that, you will be able to ask the chatbot your questions 
    in a dialog format. 
    The chatbot will take into account the most relevant materials 
    and information from the knowledge base during the conversation.
    """
)


def read_json(file_path):
    with open(file_path) as file:
        access_data = json.load(file)
    return access_data


@st.cache_resource
def initialize_faiss_vectorstore():
    """
    Vectorstore database initialization.
    
    We use FAISS instead of Chroma in this application.
    
    """
    API_CREDS = read_json(file_path='apicreds.json')
    APP_CONFIG = read_json(file_path='config.json')
    DATA_PATH = f"{APP_CONFIG['imgs_path']}/rag"
    FAISS_INDEX_PATH = "./faiss_index"

    ####################################
    ########## YOUR CODE HERE ##########
    ####################################
    # You have to create embeddings 
    # for vectorstore below
    
    embeddings = # YOUR CODE HERE

    # HINT: use code from chapter 7.3 
    # of the RAG notebook 
    
    ####################################
    
    # If index exists...
    if os.path.exists(FAISS_INDEX_PATH):
        with st.spinner('Loading FAISS index...'):
            vectorstore = FAISS.load_local(
                FAISS_INDEX_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            st.success("FAISS index loaded")
            return vectorstore, API_CREDS
    
    # ...or create new index
    with st.spinner('Documents are loading...'):
        ####################################
        ########## YOUR CODE HERE ##########
        ####################################
        # You have to create loader and docs
        # objects to load documents

        loader = # YOUR CODE HERE
        docs = # YOUR CODE HERE
        
        # HINT: use code from chapter 7.2 
        # of the RAG notebook 
        
        ####################################
    
    with st.spinner('Processing documents...'):
        ####################################
        ########## YOUR CODE HERE ##########
        ####################################
        # You have to create text splitter 
        # and get splits

        text_splitter = # YOUR CODE HERE
        splits = # YOUR CODE HERE
        
        # HINT: use code from chapter 7.2 
        # of the RAG notebook 
        
        ####################################
    
    with st.spinner('Creating FAISS index...'):
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local(FAISS_INDEX_PATH)
        st.success("FAISS index created and saved")
    
    return vectorstore, API_CREDS

def get_rag_chain(vectorstore, template, temperature, k_max, api_creds):
    """
    RAG initialization with input parameters.
    
    Args:
      :vectorstore:
      :template:
      :temperature:
      :k_max:
      :api_creds:

    Returns:
      RAG chain instance
    
    """
    ####################################
    ########## YOUR CODE HERE ##########
    ####################################
    # You have to create embeddings 
    # for vectorstore below
    
    retriever = # YOUR CODE HERE
    prompt = # YOUR CODE HERE
    llm = # YOUR CODE HERE
    rag_chain = # YOUR CODE HERE

    # HINT: use code from chapter 7.4
    # of the RAG notebook 
    
    ####################################
    
    return rag_chain


# We init vectorstore here and put in to app state
# for single load when application starts / restarts
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore, st.session_state.api_creds = initialize_faiss_vectorstore()

# Chat prompt
# You may experiment with it, 
# but keep "Context: {context}\n" and "Question: {question}\n"
default_instruction = (
    "Use the following pieces of context to answer the question at the end. "
    "If you don't know the answer, just say that you don't know, don't try to make up an answer. "
    "Use three sentences maximum and keep the answer as concise as possible. "
    "Always say \"thanks for asking!\" at the end of the answer. \n"
    "Context: {context}\n"
    "Question: {question}\n"
    "Answer: "
)

st.write('#### Give a prompt')
template = st.text_area(
    'Input prompt for chat-bot',
    default_instruction
)

st.write('#### Temperature for bot')
st.write(
    """
    The higher the value of this parameter, the more creative
    and random the model's responses will be. Accepts values
    from 0 (inclusive) to 1 (inclusive).
    Default value: 0 (no creativity)
    """
)
temperature = st.slider("Input temperature for chat-bot", .0, 1., .0, .1)

st.write('#### Enter the number of relevant documents')
st.write(
    """
    You need to specify the maximum number of documents in a single
    search to limit the search scope for the chat-bot.
    Default value: 3 documents
    """
)
k_max = st.slider('Enter the number of documents', 1, 5, 3)

# Create RAG with parameters (temperature, k_max, ...)
####################################
########## YOUR CODE HERE ##########
####################################
# You have to create embeddings 
# for vectorstore below

rag_chain = # YOUR CODE HERE

# HINT: use function `get_rag_chain(...)`
# from above and pass all parameters like
#   st.session_state.vectorstore, 
#   template, 
#   temperature, 
#   k_max, 
#   st.session_state.api_creds
# to that function

####################################

# Start chat
st.write('#### Ask chat-bot your questions')

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if query := st.chat_input('Enter your message'):
    st.chat_message('user').markdown(query)
    st.session_state.messages.append(
        {
            'role': 'user',
            'content': query
        }
    )
    
    answer = rag_chain.invoke(query)
    with st.chat_message('assistant'):
        st.markdown(answer)
    st.session_state.messages.append(
        {
            'role': 'assistant',
            'content': answer
        }
    )