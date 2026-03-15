import streamlit as st
from rag.rag_pipeline import ask_question

st.title("AI Document Knowledge Assistant")

st.write("Powered by Vector Search (Endee concept)")

query = st.text_input("Ask a question about the documents:")

if query:

    answer, context_docs = ask_question(query)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Retrieved Context")
    for doc in context_docs:
        st.write("- ", doc)