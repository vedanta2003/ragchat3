import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import tempfile
import os
from dotenv import load_dotenv

# Load environment variables

load_dotenv(override=True) 


# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = False

def process_files(uploaded_files):
    """Process uploaded files and create vector store."""
    if not uploaded_files:
        return None
    
    # Combine all text from files
    text = ""
    for file in uploaded_files:
        text += file.getvalue().decode("utf-8") + "\n\n"
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Create vector store with cosine similarity
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(
        chunks, 
        embeddings,
        collection_metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )
    
    return vectorstore

def create_conversation_chain(vectorstore):
    """Create conversation chain with memory."""
    llm = ChatOpenAI(temperature=0.7)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # Specify which output key to use for memory
    )
    
    # Configure retriever to get top 3 most relevant chunks
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}  # Get top 3 most relevant chunks
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,  # Return the source documents for transparency
        combine_docs_chain_kwargs={"prompt": None}  # Use default prompt
    )
    
    return conversation_chain

# Streamlit UI
st.title("📚 Document Q&A Chatbot")
st.write("Upload up to 5 text files and ask questions about their content!")

# File upload
uploaded_files = st.file_uploader(
    "Upload your text files",
    type=["txt"],
    accept_multiple_files=True,
    help="You can upload up to 5 text files"
)

# Process files when uploaded
if uploaded_files and len(uploaded_files) > 0 and not st.session_state.processed_files:
    if len(uploaded_files) > 5:
        st.error("Please upload a maximum of 5 files.")
    else:
        with st.spinner("Processing files..."):
            vectorstore = process_files(uploaded_files)
            st.session_state.conversation = create_conversation_chain(vectorstore)
            st.session_state.processed_files = True
            st.success("Files processed successfully! You can now ask questions.")

# Chat interface
if st.session_state.conversation is not None:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents"):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.conversation({"question": prompt})
                st.write(response["answer"])
                
                # Display source documents immediately after the response
                st.markdown("---")
                st.markdown("### 📑 Top 3 Most Relevant Chunks")
                for i, doc in enumerate(response["source_documents"], 1):
                    with st.expander(f"Chunk {i} (Most Relevant)"):
                        st.markdown(f"**Content:**")
                        st.markdown(doc.page_content)
                        st.markdown("---")
                
                # Store only the answer in chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response["answer"]
                })
else:
    st.info("Please upload some text files to begin chatting!") 