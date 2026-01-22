import sys
import os

# Go one level up from generation/ to project root
PROJECT_ROOT = os.path.abspath("..")
sys.path.insert(0, PROJECT_ROOT)

print("Project root added to path:", PROJECT_ROOT)

from embeddings.embed import final_query
def is_greeting_or_meta(query: str) -> bool:
    q = query.lower().strip()

    greetings = [
        "hi", "hello", "hey", "who are you",
        "what are you", "how are you", "what can you do"
    ]

    if any(g in q for g in greetings):
        return True

    sanskrit_greetings = [
        "नमस्ते", "कः असि", "का असि",
        "भवान् कः", "भवती का", "कथम् असि"
    ]

    if any(g in query for g in sanskrit_greetings):
        return True

    return False

def greeting_response(query: str) -> str:
    if any(word in query.lower() for word in ["hi", "hello", "hey"]):
       return (
        "नमस्ते। अहं संस्कृत-आधारितः दस्तावेज-आश्रितः "
        "Retrieval-Augmented Generation (RAG) सहायः अस्मि। "
        "अहं केवलं प्रदत्तेषु संस्कृत-दस्तावेजेषु आधारितान् "
        "प्रश्नान् एव उत्तरितुं शक्नोमि।"
    )

    if "नमस्ते" in query:
        return (
            "नमस्ते। अहं संस्कृत-आधारितः RAG सहायः अस्मि। "
            "प्रदत्तेषु संस्कृत-दस्तावेजेषु आधारितान् प्रश्नान् एव उत्तरितुं शक्नोमि।"
        )

    return (
        "I am a Sanskrit document-based chatbot. "
        "Please ask questions related to the provided documents."
    )



def build_context(retrieved_chunks):
    """
    Input: list of dicts from final_query()
    Output: clean context string
    """
    context_blocks = []

    for chunk in retrieved_chunks:
        context_blocks.append(chunk["text"])

    return "\n\n".join(context_blocks)

def build_prompt(query, context):
    return f"""
त्वं संस्कृत-आधारितः Retrieval-Augmented Generation (RAG) सहायः असि।

तव कार्यम्— अधोलिखितात् प्रदत्त-सन्दर्भात् (Context) एव
उपयोक्तृप्रश्नस्य यथार्थं सुसम्बद्धं च उत्तरं दातुम्।

नियमाः—
1. केवलं प्रदत्त-सन्दर्भस्य आधारेण एव उत्तरं देहि।
2. स्वकल्पितं, बाह्यं, असन्दर्भितं वा ज्ञानं न प्रयोजय।
3. यदि सन्दर्भे उत्तरं न लभ्यते, तर्हि स्पष्टं वद—
   "दत्तसन्दर्भे उत्तरं न उपलब्धम्।"
4. यदि प्रश्नः संस्कृते वा लिप्यन्तरितसंस्कृते अस्ति,
   तर्हि उत्तरं केवलं संस्कृते एव देहि।
5. यदि प्रश्नः स्पष्टतया आङ्ग्लभाषायां (English) अस्ति,
   तर्हि उत्तरं आङ्ग्लभाषायां एव देहि।
6. उत्तरं संक्षिप्तं, स्पष्टं, व्याकरणशुद्धं च भवेत्।

सन्दर्भः:
----------------
{context}
----------------

प्रश्नः:
{query}

उत्तरम्:
"""

import requests
import json

def generate_with_ollama(prompt, model="gemma3:4b"):
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()

    return response.json()["response"]

def rag_chat(query: str) -> str:
    # 0. Greeting / meta check
    if is_greeting_or_meta(query):
        return greeting_response(query)

    # 1. Retrieval
    retrieved = final_query(query)

    if not retrieved:
        return "दत्तसन्दर्भे उत्तरं न उपलब्धम्।"

    # 2. Build context
    context = "\n\n".join(r["text"] for r in retrieved)

    # 3. Build prompt
    prompt = build_prompt(query, context)

    # 4. Generate answer
    return generate_with_ollama(prompt)

