import os
import dotenv
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('api_key')
import openai
import streamlit as st

from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

from embeddings import encode_text
from embeddings import calculate_similarity

import numpy as np

st.set_page_config(
        page_title="RAG Implemented",
    )
st.header("Ask questions based on your own documents!")
pwd = "/Users/sohumfuke/Desktop/Programming2/NLP_Project2"


def create_bert_embedding(data):
    """Use the encode_text function to encode documents in the directory data"""
    docs = []
    files = os.listdir(pwd+"/"+data)
    for file in files:
    # Load the document, split it into chunks, embed each chunk and load it into the vector store.
        raw_documents = TextLoader(data+"/"+file).load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        documents = text_splitter.split_documents(raw_documents)
        docs.extend(documents)

    embeddings_store = {}
    for doc in docs:
        embeddings_store[tuple(encode_text(doc.page_content))] = doc.page_content
    return embeddings_store

def create_embedding(data):
    """Creates embeddings of documents present in the data directory"""
    # Load the document, split it into chunks, embed each chunk and load it into the vector store.    
    docs = []
    files = os.listdir(pwd+"/"+data)
    for file in files:
    # Load the document, split it into chunks, embed each chunk and load it into the vector store.
        raw_documents = TextLoader(data+"/"+file).load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        documents = text_splitter.split_documents(raw_documents)
        docs.extend(documents)
    db = Chroma.from_documents(docs, OpenAIEmbeddings())
    # db = Chroma.from_documents(docs, encode_text())

    return db

def retrieve(prompt, n=3):
    """Retrieve the n most relevant chunks to the prompt"""
    p_emb = encode_text(prompt)
    chunks_ranked = {}
    for i in st.session_state.db.keys():
        sim = calculate_similarity(np.array(i), p_emb)
        chunks_ranked[sim] = st.session_state.db[i]

    sorted_chunks_ranked = {key: chunks_ranked[key] for key in sorted(chunks_ranked)}
    # first_key, first_value = next(iter(sorted_chunks_ranked.items()))
    # print(first_value)
    first_n_chunks = {k: sorted_chunks_ranked[k] for k in list(sorted_chunks_ranked)[:n]}


    # print(list(first_n_chunks.values()))
    return first_n_chunks.values()

def get_response(docs, prompt):
    """Use GPT 3.5 to get a response given prompt and relevant chunks"""
    message_list = []
    context = ""
    for doc in docs:
        context += doc.page_content
        context += "\n\n"
    full_prompt = f"Given the following context information:\n{context}\nanswer the following question\n{prompt}"
    user_message = {"role": "user", "content": full_prompt}
    message_list.append(user_message)

    
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=message_list,
        temperature=0.7
    )

    return response.choices[0].message.content

if "db" not in st.session_state:
    st.session_state.db = None


### Upload Documents
link_input = st.sidebar.empty()
with link_input.form("my-form", clear_on_submit=True):
    files = st.file_uploader("Create Index", accept_multiple_files=True)
    create_new = st.form_submit_button("Generate")

    if create_new and files is not None:
        with st.spinner('Creating index. This may take a few seconds...'):
            my_bar = st.progress(0.0)

            ### Deleting Existing Files
            old_files = os.listdir(pwd+"/data")
            for file in old_files:
                os.remove(pwd + "/data/" + file)
            my_bar.progress(0.2)
            ### Downloading Files
            for file in files:
                bytes_data = file.getvalue()
                file_name = file.name
                save_path = pwd + "/data/" + file_name
                with open(save_path, "wb") as f:
                    f.write(bytes_data)
            my_bar.progress(0.8)
            ## Download Over
            my_bar.empty()

            st.session_state.db = create_embedding("data")
    


if "messages" not in st.session_state or st.sidebar.button("Clear Chat"):
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

### User Input
if prompt := st.chat_input("Ask Me!"):
    st.chat_message("user").markdown(prompt)
    with st.spinner("Answering..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        docs = st.session_state.db.similarity_search(prompt, k=3)
        # docs = retrieve(prompt)
        
        response = get_response(docs, prompt)
        st.chat_message("assistant").markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})