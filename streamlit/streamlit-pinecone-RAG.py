# streamlit run streamlit-pinecone-RAG.py

import os
import pinecone
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone

# from langchain.llms import OpenAI
# from langchain.llms.openai import OpenAIChat
from langchain.chat_models import ChatOpenAI
# from langchain.document_loaders import DirectoryLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

import streamlit as st

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv('../.env')) # read local .env file

OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_INDEX = os.getenv('PINECONE_INDEX')

# initialize pinecone client and embeddings
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vector_store = Pinecone.from_existing_index(PINECONE_INDEX, embeddings)

st.title("Lab protocol chatbot")


def query_llm(retriever, query):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY),
        retriever=retriever,
        return_source_documents=True,
    )
    result = qa_chain({'question': query, 'chat_history': st.session_state.messages})
    result = result['answer']
    st.session_state.messages.append((query, result))
    return result

def boot():

    if "messages" not in st.session_state:
        st.session_state.messages = []    
    #

    msgs = StreamlitChatMessageHistory()  

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=msgs,
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0, streaming=True),
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    st.session_state.retriever = vector_store.as_retriever()

    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])    
    #
    if query := st.chat_input():
        st.chat_message("human").write(query)
        response = query_llm(st.session_state.retriever, query)
        st.chat_message("ai").write(response)

if __name__ == '__main__':
    #
    boot()