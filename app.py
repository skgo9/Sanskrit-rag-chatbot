import streamlit as st
import time

# Import your RAG function
from context import rag_chat   # adjust import if filename differs

# Page config
st.set_page_config(
    page_title="Sanskrit RAG Chatbot",
    page_icon="📜",
    layout="centered"
)

st.title("📜 Sanskrit RAG Chatbot")
st.caption("CPU-based • Local • Context-aware Sanskrit Question Answering")

st.markdown(
    """
    **Instructions:**
    - Ask questions in **Sanskrit** or **English**
    - Sanskrit queries → Sanskrit answers
    - English queries → English answers
    - Answers are generated **only from provided Sanskrit documents**
    """
)

# User input
query = st.text_area(
    "Enter your question:",
    placeholder="मूर्खभृत्यस्य कथां संक्षेपेण कथय",
    height=100
)

# Ask button
if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking... (CPU-based inference)"):
            start_time = time.time()
            try:
                response = rag_chat(query)
                elapsed = time.time() - start_time
            except Exception as e:
                st.error(f"Error: {e}")
                response = None

        if response:
            st.subheader("Answer")
            st.write(response)

            st.caption(f"⏱ Response time: {elapsed:.2f} seconds")
