"""Generation Node using Gemini 2.5 Flash for strictly grounded RAG answers with streaming support."""

from typing import Any, Dict, List
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from src.state import GraphState
from src.utils.logger import logger
from config.settings import settings


def generate_node(state: GraphState) -> GraphState:
    """LangGraph node that generates the final grounded answer using Gemini.

    Compatible with both synchronous execution (rag_app.invoke) and asynchronous
    token streaming (rag_app.astream_events).
    """
    question = state.get("question", "")
    documents = state.get("documents", [])
    chat_history = list(state.get("chat_history", []))

    if not documents:
        generation = "Maaf, saya tidak menjumpai maklumat yang berkaitan dalam FAQ."
        updated_history = list(chat_history) + [
            HumanMessage(content=question),
            AIMessage(content=generation),
        ]
        return {
            **state,
            "generation": generation,
            "chat_history": updated_history,
            "sources": [],
        }

    context_str = ""
    sources: List[Dict[str, Any]] = []

    for idx, doc in enumerate(documents, 1):
        faq_id = doc.metadata.get("id", "N/A")
        faq_question = doc.metadata.get("question", doc.page_content)
        faq_answer = doc.metadata.get("answer", "")

        context_str += f"\n[Dokumen {idx} | ID: {faq_id}]\n"
        context_str += f"Soalan: {faq_question}\nJawapan:\n{faq_answer}\n"
        sources.append(doc.metadata)

    try:
        api_key = settings.effective_api_key
        llm_kwargs = {
            "model": settings.gemini_model,
            "temperature": 0.0,
        }
        if api_key:
            llm_kwargs["google_api_key"] = api_key
        llm = ChatGoogleGenerativeAI(**llm_kwargs)
    except Exception as e:
        logger.error(f"Failed to initialize ChatGoogleGenerativeAI: {e}")
        generation = f"Ralat sistem: {e}"
        updated_history = list(chat_history) + [
            HumanMessage(content=question),
            AIMessage(content=generation),
        ]
        return {
            **state,
            "generation": generation,
            "chat_history": updated_history,
            "sources": [],
        }

    system_prompt = f"""You are a helpful and polite customer support assistant for Tonton (Media Prima).
Your task is to answer the user's question accurately and helpfully based on the provided FAQ CONTEXT below.

FAQ CONTEXT:
{context_str}

RULES:
1. Ground your answers strictly in the FAQ CONTEXT.
2. If the user's question relates to a topic present in the FAQ (e.g. how to subscribe, upgrade to TontonUp, account reset, cancellations, payment issues, ads), extract and explain the relevant steps, guidance, or official website URLs (e.g. https://www.tonton.com.my/tontonup) provided in the context.
3. Only if the provided context is completely irrelevant to the question, state politely that you do not have specific information on that topic in the FAQ and advise them to contact support.
4. Answer in the same language as the user's question (Bahasa Melayu or English).
5. If there are step-by-step instructions in the FAQ, format them clearly using numbered lists or bullet points.
6. If there is a URL link in the FAQ, include it directly in your response.
7. Keep your tone polite, professional, and clear. Do NOT wrap your whole response in backticks or code blocks.
"""

    messages = [SystemMessage(content=system_prompt)]
    if chat_history:
        messages.extend(chat_history)
    messages.append(HumanMessage(content=question))

    logger.info("Invoking Gemini for answer generation & streaming...")
    try:
        response = llm.invoke(messages)
        generation = response.content
        if isinstance(generation, str):
            generation = generation.strip()
            # Clean any stray unclosed markdown code fences
            if generation.startswith("```") and not generation.endswith("```"):
                generation = generation.lstrip("`").strip()
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        generation = "Maaf, ralat berlaku semasa menjana jawapan. Sila cuba sebentar lagi."
        sources = []

    updated_history = list(chat_history) + [
        HumanMessage(content=question),
        AIMessage(content=generation),
    ]

    return {
        **state,
        "generation": generation,
        "chat_history": updated_history,
        "sources": sources,
    }
