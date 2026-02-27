import json
import os

import streamlit as st
from ollama import ResponseError as OllamaResponseError
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app_config import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    get_python_compatibility_message,
    resolve_ollama_settings,
)

try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    import pyttsx3
except Exception:
    pyttsx3 = None


HISTORY_DIR = "chat_history"
os.makedirs(HISTORY_DIR, exist_ok=True)


def get_history_filepath():
    session_id = "chat1"
    return os.path.join(HISTORY_DIR, f"{session_id}.json")


def load_history():
    filepath = get_history_filepath()
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as file_handle:
            return json.load(file_handle)
    return []


def save_history(messages):
    filepath = get_history_filepath()
    with open(filepath, "w", encoding="utf-8") as file_handle:
        payload = [{"type": msg.type, "content": msg.content} for msg in messages]
        json.dump(payload, file_handle, indent=2)


def init_voice_engine():
    if pyttsx3 is None:
        return None

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        return engine
    except Exception:
        return None


def speak(text, voice_engine):
    if not voice_engine:
        st.warning("Voice output is unavailable on this machine.")
        return

    try:
        voice_engine.say(text)
        voice_engine.runAndWait()
    except Exception as error:
        st.warning(f"Voice output failed: {error}")


def recognize_speech():
    if sr is None:
        st.warning("SpeechRecognition is not installed, so voice input is disabled.")
        return ""

    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            st.info("Listening...")
            audio = recognizer.listen(source)
    except Exception as error:
        st.warning(f"Microphone is unavailable: {error}")
        return ""

    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        st.warning("Could not understand the audio.")
    except sr.RequestError as error:
        st.warning(f"Speech recognition service failed: {error}")

    return ""


def build_retriever(uploaded_file, embedding_model, embedding_base_url):
    extension = os.path.splitext(uploaded_file.name)[1].lower()
    local_filename = f"uploaded_doc{extension}"

    with open(local_filename, "wb") as file_handle:
        file_handle.write(uploaded_file.read())

    try:
        if extension == ".pdf":
            loader = PyPDFLoader(local_filename)
        else:
            loader = TextLoader(local_filename, encoding="utf-8")

        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        if not chunks:
            raise ValueError("No document text could be extracted from the uploaded file.")

        embeddings = OllamaEmbeddings(model=embedding_model, base_url=embedding_base_url)
        db = FAISS.from_documents(chunks, embeddings)
        return db.as_retriever(search_kwargs={"k": 3})
    finally:
        if os.path.exists(local_filename):
            os.remove(local_filename)


st.set_page_config(page_title="Local Chatbot", page_icon="")
st.title("Daniel's ChatBot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()
    st.session_state.messages = []

    for msg in load_history():
        if msg["type"] == "human":
            st.session_state.chat_history.add_user_message(msg["content"])
            st.session_state.messages.append(HumanMessage(content=msg["content"]))
        elif msg["type"] == "ai":
            st.session_state.chat_history.add_ai_message(msg["content"])
            st.session_state.messages.append(AIMessage(content=msg["content"]))

if "retriever" not in st.session_state:
    st.session_state.retriever = None
    st.session_state.retriever_key = None

if "voice_engine" not in st.session_state:
    st.session_state.voice_engine = init_voice_engine()

python_warning = get_python_compatibility_message()

with st.sidebar:
    st.subheader("Assistant Settings")

    if python_warning:
        st.warning(python_warning)

    system_prompt = st.text_area(
        "System prompt",
        value="You are a helpful, witty, concise assistant.",
    )

    ollama_target = st.radio("Chat Ollama target", ["Local", "Remote"], horizontal=True)

    remote_url = ""
    if ollama_target == "Remote":
        remote_url = st.text_input(
            "Remote Ollama URL",
            value=os.getenv("OLLAMA_REMOTE_URL", ""),
            placeholder="https://your-remote-ollama-host",
        )

    chat_model = st.text_input("Chat model", value=DEFAULT_CHAT_MODEL)
    embedding_target = st.selectbox(
        "Embedding target",
        ["Same as chat", "Local"],
        help="Use local embeddings when the chat model is remote/cloud-only.",
    )
    embedding_model = st.text_input(
        "Embedding model",
        value=DEFAULT_EMBEDDING_MODEL,
        help="Use an embedding model that exists on the selected Ollama endpoint.",
    )

    use_voice_input = st.checkbox(
        "Use voice input",
        disabled=sr is None,
        help="Install SpeechRecognition and microphone drivers to enable.",
    )
    use_voice_output = st.checkbox(
        "Use voice output",
        disabled=st.session_state.voice_engine is None,
        help="Install pyttsx3 and local speech engine to enable.",
    )

ollama_settings = resolve_ollama_settings(
    target=ollama_target,
    remote_url=remote_url,
    chat_model=chat_model,
    embedding_model=embedding_model,
)

if embedding_target == "Local":
    embedding_base_url = resolve_ollama_settings(target="local").base_url
else:
    embedding_base_url = ollama_settings.base_url

st.caption(
    f"Chat: {ollama_settings.target} @ {ollama_settings.base_url} ({ollama_settings.chat_model}) | "
    f"Embeddings: {embedding_target} @ {embedding_base_url} ({ollama_settings.embedding_model})"
)

llm = ChatOllama(model=ollama_settings.chat_model, base_url=ollama_settings.base_url)

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file is not None:
    retriever_key = (
        uploaded_file.name,
        embedding_base_url,
        ollama_settings.embedding_model,
    )

    if st.session_state.retriever_key != retriever_key:
        with st.spinner("Indexing document..."):
            try:
                st.session_state.retriever = build_retriever(
                    uploaded_file,
                    ollama_settings.embedding_model,
                    embedding_base_url,
                )
                st.session_state.retriever_key = retriever_key
                st.success(
                    f"Document indexed with embedding model '{ollama_settings.embedding_model}' at {embedding_base_url}."
                )
            except OllamaResponseError as error:
                st.session_state.retriever = None
                st.session_state.retriever_key = None
                st.error(
                    "Document indexing failed because the embedding model is unavailable on this Ollama target."
                )
                st.info(
                    f"Workaround: switch 'Embedding model' to '{DEFAULT_EMBEDDING_MODEL}' or another available embedding model. "
                    "If chat uses a remote cloud model, set 'Embedding target' to Local."
                )
                st.caption(f"Ollama error: {error}")
            except Exception as error:
                st.session_state.retriever = None
                st.session_state.retriever_key = None
                st.error(f"Document processing failed: {error}")

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)

user_input = None
if use_voice_input:
    if st.button("Speak"):
        user_input = recognize_speech()
else:
    user_input = st.chat_input("Ask me anything...")

if user_input:
    st.chat_message("user").write(user_input)

    llm_input = user_input
    if st.session_state.retriever:
        try:
            context_docs = st.session_state.retriever.invoke(user_input)
            context_text = "\n\n".join(doc.page_content for doc in context_docs[:3])
            llm_input = (
                "Use the following document context when it is relevant. "
                "If the answer is not in the context, say so clearly.\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {user_input}"
            )
        except Exception as error:
            st.warning(f"Document context lookup failed, falling back to normal chat: {error}")

    prompt_messages = [SystemMessage(content=system_prompt)]
    prompt_messages.extend(st.session_state.chat_history.messages)
    prompt_messages.append(HumanMessage(content=llm_input))

    try:
        response = llm.invoke(prompt_messages)
    except OllamaResponseError as error:
        st.error(f"Chat request failed: {error}")
        if "not found" in str(error).lower():
            st.info(
                "Workaround: verify the selected chat model exists on the current Ollama target or switch to a model that is already available."
            )
    except Exception as error:
        st.error(f"Unexpected chat error: {error}")
    else:
        st.session_state.chat_history.add_user_message(user_input)
        st.session_state.chat_history.add_ai_message(response.content)
        st.session_state.messages.append(HumanMessage(content=user_input))
        st.session_state.messages.append(AIMessage(content=response.content))
        save_history(st.session_state.messages)

        st.chat_message("assistant").write(response.content)

        if use_voice_output:
            speak(response.content, st.session_state.voice_engine)
