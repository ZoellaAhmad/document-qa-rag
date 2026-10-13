import hashlib
import streamlit as st
from rag import build_index, answer_question

st.title("Document Q&A")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()

    # Only re-index when a different file is uploaded
    if st.session_state.get("indexed_file") != file_hash:
        with st.spinner("Indexing document..."):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            build_index("temp.pdf")
        st.session_state["indexed_file"] = file_hash

    st.success(f"Indexed: {uploaded_file.name}. Ask away.")

    question = st.text_input("Ask a question about the document:")
    if question:
        with st.spinner("Thinking..."):
            answer = answer_question(question)
        st.write(answer)
