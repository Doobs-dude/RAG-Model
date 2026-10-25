import streamlit as st
from sentence_transformers import CrossEncoder

# Import from src
from src.data_loader import load_metadata
from src.embeddings import load_embedding_model, load_faiss_index, get_device
from src.retriever import Retriever
from src.rag_chain import RAGChain

st.set_page_config(page_title="Academic RAG Weapon", page_icon="📚", layout="wide")
st.title("📚 Academic RAG Weapon")
st.caption("Use a very scientific dataset and AI to answer questions about Comp Sci effectively!")

@st.cache_resource
def initialize_system():
    device = get_device()
    
    # Load dependencies
    bi_encoder = load_embedding_model(device=device)
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
    index = load_faiss_index("vector_store/arxiv_full.index")
    metadata_df = load_metadata("vector_store/arxiv_full_metadata.pkl")
    
    # Keep it modular
    retriever = Retriever(bi_encoder, reranker, index, metadata_df)
    rag_chain = RAGChain()
    
    return retriever, rag_chain

retriever, rag_chain = initialize_system()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me any question related to Comp Sci, I'm pretty accurate."}
    ]

# Render History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Handling
if prompt := st.chat_input("Ask a question about computer science"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving papers & building answer..."):
            try:
                top_results = retriever.search_and_rerank(prompt)
                refined_answer = rag_chain.run_2_pass_generation(prompt, top_results)
                
                st.markdown(refined_answer)
                
                with st.expander("View Retrieved Sources"):
                    for idx, row in top_results.reset_index().iterrows():
                        st.markdown(f"**[{idx+1}] {row['title']}**")
                        st.caption(f"Authors: {row['authors']} | Category: {row['category']}")
                        st.write(row['summary'])
                
                st.session_state.messages.append({"role": "assistant", "content": refined_answer})
            except Exception as e:
                st.error(f"Error processing request: {e}")