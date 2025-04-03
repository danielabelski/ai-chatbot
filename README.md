# 🧠 Personal AI Chatbot with Document Q&A and Voice Support

A fully local, privacy-focused AI chatbot built with **LangChain**, **Ollama**, and **Streamlit**, designed to function as a personal assistant with voice interaction and document understanding.

## 🔧 Features

- **LLM-Powered Chat**  
  Runs completely offline using local models (e.g., `mistral:7b`) via [Ollama](https://ollama.com).

- **Streamlit Interface**  
  A modern web chat UI with avatars, personality configuration, and persistent chat memory.

- **📄 Document Q&A**  
  Upload `.pdf` or `.txt` files and ask questions based on their content using a FAISS-powered retriever.

- **💾 Persistent Chat History**  
  Saves chat sessions to disk and reloads them automatically.

- **🎙️ Voice Input & Output**  
  Use your microphone for speech-to-text and hear the bot reply via text-to-speech. Toggle from the sidebar.

- **🧠 Configurable Personality**  
  Customise the assistant’s tone and behavior via a system prompt field in the sidebar.

---

## 🛠 Tech Stack

- Python
  - [LangChain](https://python.langchain.com/)
  - [Streamlit](https://streamlit.io/)
  - [FAISS](https://github.com/facebookresearch/faiss)
  - [PyPDFLoader](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf)
  - [SpeechRecognition](https://pypi.org/project/SpeechRecognition/)
  - [pyttsx3](https://pypi.org/project/pyttsx3/)
- [Ollama](https://ollama.com) 

---

## 🚀 Getting Started

```bash
# Clone and install requirements
pip install -r requirements.txt

# Run the chatbot
streamlit run app.py
