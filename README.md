# Personal AI Chatbot (Local + Ollama Remote Option)

Privacy-focused chatbot with Streamlit UI and terminal mode, built with LangChain + Ollama. Supports normal chat, document Q&A (PDF/TXT), chat persistence, and optional voice input/output.

## Compatibility

- Python: tested on `3.10` and `3.11`
- Python `> 3.11` is not officially supported yet because some dependencies may fail to install or run consistently

## Features

- Chat with local or remote Ollama from one configuration flow
- Streamlit UI (`UIChat.py`) and terminal chat (`TerminalChat.py`)
- PDF/TXT upload and retrieval-augmented responses (FAISS)
- Persistent chat history saved to `chat_history/chat1.json`
- Optional voice input (SpeechRecognition) and voice output (pyttsx3)

## Install

```bash
pip install -r requirements.txt
```

## Run

### Streamlit UI

```bash
streamlit run UIChat.py
```

In the sidebar you can set:
- `Chat Ollama target`: `Local` or `Remote`
- `Remote Ollama URL` (when target is `Remote`)
- `Chat model`
- `Embedding target`: `Same as chat` or `Local`
- `Embedding model`

### Terminal Mode

```bash
python TerminalChat.py --target local --model llama3.2:3b
```

Remote example:

```bash
python TerminalChat.py --target remote --remote-url https://your-remote-ollama-host --model llama3.2:3b
```

## Cloud Model Upload Workaround

If chat works but PDF/TXT upload fails with an error like:
`model "..." not found, try pulling it first`

This usually means the selected embedding model is not available on the current Ollama endpoint.

Use this workaround:
- Keep your chat model as-is (for example cloud-hosted)
- Set `Embedding model` to a known embedding model (default is `nomic-embed-text`)
- Set `Embedding target` to `Local` when the remote endpoint does not provide embedding models

## Audio Notes

Audio in UI is optional and now fails gracefully:
- If microphone or speech dependencies are unavailable, the app shows warnings instead of crashing
- Voice output is skipped with a warning when local TTS is not available
