import streamlit as st
from rag import build_index, answer_question


st.title("Document Q&A")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    build_index("temp.pdf")
    st.success("Document indexed! Ask away.")


    question = st.text_input("Ask a question about the document:")
    if question:
        answer = answer_question(question)
        st.write(answer)

