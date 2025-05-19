import boto3
import streamlit as st
import os
import shutil
from pathlib import Path
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Prompt Template
prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use at least summarize with 
250 words with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

# Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime", region_name="ap-south-1")

# Get embedding model
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)

# Load documents from /data folder
def get_documents():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    docs = text_splitter.split_documents(documents)
    return docs

# Create and save FAISS vector store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

# Load Bedrock LLM
def get_llm():
    llm = Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock,
                  model_kwargs={'max_gen_len': 512})
    return llm

# Prompt Template Setup
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# QA Chain for RAG
def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

# Clear previous files before saving new ones
def clear_data_folder():
    folder = "data"
    os.makedirs(folder, exist_ok=True)
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # delete file or link
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


# Save uploaded files to /data folder
def save_uploaded_files(uploaded_files):
    clear_data_folder()
    for file in uploaded_files:
        file_path = Path("data") / file.name
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

# Streamlit UI
def main():
    st.set_page_config(page_title="RAG Demo")
    st.header("RAG Application")
    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Upload & Create Vector Store")
        uploaded_files = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)

        if uploaded_files and st.button("Store Vector"):
            with st.spinner("Processing uploaded PDFs..."):
                save_uploaded_files(uploaded_files)
                docs = get_documents()
                get_vector_store(docs)
                st.success("Vector store created successfully!")

    if st.button("Send") and user_question.strip() != "":
        with st.spinner("Processing your question..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_llm()
            st.markdown("### Answer")
            st.write(get_response_llm(llm, faiss_index, user_question))


if __name__ == "__main__":
    main()
