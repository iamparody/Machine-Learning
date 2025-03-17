import chainlit as cl
import ollama
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# Load FAQs from kra_faqs.json
try:
    with open("kra_faqs.json", "r", encoding="utf-8") as f:
        faq_data = json.load(f)
    faq_list = faq_data.get("general", [])
except Exception as e:
    logger.error(f"Error loading FAQs: {e}")
    faq_list = []

# Normalize questions and map answers
def normalize_text(text):
    return " ".join(text.lower().strip().split())

questions = [normalize_text(item["question"]) for item in faq_list]
answers = {normalize_text(item["question"]): item["answer"] for item in faq_list}

# Load or create embeddings
def load_or_create_embeddings(file_path, texts):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    embeddings = embed_model.encode(texts, convert_to_numpy=True)
    with open(file_path, "wb") as f:
        pickle.dump(embeddings, f)
    return embeddings

question_embeddings = load_or_create_embeddings("faq_embeddings.pkl", questions)
faq_index = faiss.IndexFlatIP(question_embeddings.shape[1])
faq_index.add(question_embeddings)

SIMILARITY_THRESHOLD = 0.6  # Kept higher to favor fallback in FAQ check

def get_faq_answer(user_query):
    """Find the best FAQ match or return None if below threshold."""
    normalized_query = normalize_text(user_query)
    user_embedding = embed_model.encode([normalized_query], convert_to_numpy=True)
    D, I = faq_index.search(user_embedding, 3)

    query_keywords = set(normalized_query.split())
    best_match_idx = I[0][0]
    best_score = D[0][0]

    key_phrases = ["tax amnesty", "qualify"]
    for i in range(min(3, len(D[0]))):
        faq_text = questions[I[0][i]]
        faq_keywords = set(faq_text.split())
        overlap = len(query_keywords & faq_keywords) / len(query_keywords)
        boost = 0.3 if any(phrase in faq_text for phrase in key_phrases) else 0.2
        adjusted_score = D[0][i] + (overlap * boost)
        logger.info(f"Match {i+1}: {faq_text} (Original: {D[0][i]}, Adjusted: {adjusted_score})")
        if adjusted_score > best_score:
            best_score = adjusted_score
            best_match_idx = I[0][i]

    logger.info(f"User Query: {normalized_query}")
    logger.info(f"Best Match: {questions[best_match_idx]} (Final Score: {best_score})")

    if best_score > SIMILARITY_THRESHOLD:
        return answers[questions[best_match_idx]]
    return None

async def chat_with_ai(user_input, msg):
    """Try to get an answer from Ollama."""
    try:
        system_prompt = """You are a Kenyan Revenue Authority (KRA) assistant.
        Only provide responses related to KRA tax regulations, customs, and Single Customs Territory (SCT).
        If the question is unclear or outside KRA, ask for clarification or say you don’t have enough info.
        DO NOT provide information on SCP, South African Customs, or unrelated tax laws."""
        response = ollama.chat(model="llama3.2:3b", messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ], stream=True)

        full_response = ""
        for chunk in response:
            if 'message' in chunk and 'content' in chunk['message']:
                content = chunk['message']['content']
                full_response += content
                await msg.stream_token(content)

        # Check if response is useful (basic heuristic)
        if "I don’t have enough info" in full_response or "please clarify" in full_response.lower() or len(full_response.strip()) < 20:
            return False  # Signal to fall back to FAQs
        return True  # Response was good, stop here

    except Exception as e:
        logger.error(f"Ollama error: {e}")
        await msg.stream_token(f"Sorry, I encountered an issue: {str(e)}. Checking FAQs instead...\n\n")
        return False  # Fall back to FAQs on error

@cl.on_message
async def main(message: cl.Message):
    msg = cl.Message(content="")
    await msg.send()

    if message.content.lower().strip() in ["hello", "hi", "jambo"]:
        await msg.stream_token("Jambo! Welcome to the Kenyan Revenue Authority (KRA) assistant. How can I assist you today?")
    else:
        # Try Ollama first
        ollama_success = await chat_with_ai(message.content, msg)
        
        # If Ollama fails or gives a weak response, check FAQs
        if not ollama_success:
            faq_answer = get_faq_answer(message.content)
            if faq_answer:
                await msg.stream_token(f"Based on KRA FAQs:\n{faq_answer}")
            else:
                await msg.stream_token(f"No FAQ match. Checking KRA resources...\n\nFor more info, visit: https://www.kra.go.ke/helping-tax-payers/faqs/")

    await msg.update()