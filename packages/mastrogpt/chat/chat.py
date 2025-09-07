import os
import json
import time
import socket
import requests

# ------------------------------------------------------------------ #
# CONFIGURATION – edit only these lines                              #
# ------------------------------------------------------------------ #
DEFAULT_MODEL = "openai/gpt-oss-120b:free"
STATIC_MODELS = [
    "moonshotai/kimi-k2",
    "deepseek/deepseek-chat-v3.1",
    "qwen/qwen3-coder",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "openchat/openchat-7b",
    "openai/gpt-5",
    "mistralai/mistral-7b-instruct",
    "deepseek/deepseek-chat-v3.1:free",
    "openai/gpt-oss-120b:free"
]
STREAM_ENABLED = True        # False → blocking / non-streaming

# ------------------------------------------------------------------ #
# INTERNAL HELPERS                                                   #
# ------------------------------------------------------------------ #
def _get_base_url(args: dict) -> str:
    """
    Returns the base URL (with no trailing slash).
    Empty string triggers the error message in chat().
    """
    base = args.get("OLLAMA_API_HOST", os.getenv("OLLAMA_API_HOST", ""))
    return base.rstrip("/") if base else ""

def _get_headers(args: dict) -> dict:
    token = args.get("OLLAMA_API_SECRET", os.getenv("OLLAMA_API_SECRET", ""))
    hdr = {"Content-Type": "application/json"}
    if token:
        hdr["Authorization"] = f"Bearer {token}"
    return hdr

# ------------------------------------------------------------------ #
# SOCKET HELPERS                                                     #
# ------------------------------------------------------------------ #
def _open_socket(args: dict):
    host = args.get("STREAM_HOST", "").strip()
    port = int(args.get("STREAM_PORT", "0"))
    if not host or port == 0:
        return None
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    return s

def _close_socket(sock):
    try:
        sock.close()
    except Exception:
        pass

def _send(sock, payload: dict):
    if sock is None:
        return
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    sock.sendall(data)

def _stream_text(sock, text: str, chunk_size: int = 10):
    """
    Helper to stream text through socket in chunks
    """
    if sock is None:
        return
        
    # Ensure text is properly encoded
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='replace')
    
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        _send(sock, {"output": chunk})
        time.sleep(0.01)  # Small delay to simulate streaming


# ------------------------------------------------------------------ #
# MAIN ENTRY POINT                                                   #
# ------------------------------------------------------------------ #
def chat(args: dict) -> dict:
    # Initialize state from args or default
    state_input = args.get("state", {})
    
    # Handle case where state might be a JSON string
    if isinstance(state_input, str):
        try:
            state_payload = json.loads(state_input)
        except json.JSONDecodeError:
            state_payload = {}
    else:
        state_payload = state_input.copy() if isinstance(state_input, dict) else {}
    
    # Ensure state has required fields
    if "model" not in state_payload:
        state_payload["model"] = DEFAULT_MODEL
    if "history" not in state_payload:
        state_payload["history"] = []

    user_inp = str(args.get("input", "")).strip()
    
    # Handle empty input
    if not user_inp:
        usage = (
            "Welcome to the assistant.\n"
            "Type @ to list curated models.\n"
            f"Type @<prefix> to select one (default: {DEFAULT_MODEL}).\n"
        )
        return {"output": usage, "streaming": True, "state": state_payload}

    # Handle model selection commands
    if user_inp.startswith('@'):
        sock = _open_socket(args) if STREAM_ENABLED else None
        
        if user_inp == '@':
            # List available models
            model_list = "\n".join(STATIC_MODELS)
            output = f"Available models:\n{model_list}\n\nCurrent model: {state_payload['model']}"
            
            if STREAM_ENABLED:
                _send(sock, {"state": state_payload})
                _stream_text(sock, output)
                _close_socket(sock)
            
            return {"output": output, "streaming": STREAM_ENABLED, "state": state_payload}
        else:
            # Try to match model prefix
            prefix = user_inp[1:].lower()
            matched_models = [model for model in STATIC_MODELS if model.lower() == prefix]
            
            if not matched_models:
                output = f"No model found with prefix '{prefix}'. Current model remains: {state_payload['model']}"
                
                if STREAM_ENABLED:
                    _send(sock, {"state": state_payload})
                    _stream_text(sock, output)
                    _close_socket(sock)
                
                return {"output": output, "streaming": STREAM_ENABLED, "state": state_payload}
            
            if len(matched_models) > 1:
                output = f"Multiple models match prefix '{prefix}':\n" + "\n".join(matched_models)
                
                if STREAM_ENABLED:
                    _send(sock, {"state": state_payload})
                    _stream_text(sock, output)
                    _close_socket(sock)
                
                return {"output": output, "streaming": STREAM_ENABLED, "state": state_payload}
            
            # Update model in state
            state_payload["model"] = matched_models[0]
            output = f"Model updated to: {matched_models[0]}"
            
            if STREAM_ENABLED:
                _send(sock, {"state": state_payload})
                _stream_text(sock, output)
                _close_socket(sock)
            
            return {"output": output, "streaming": STREAM_ENABLED, "state": state_payload}

    # Handle normal input - prepare messages history
    messages = state_payload["history"].copy()
    messages.append({"role": "user", "content": user_inp})

    # Prepare API request
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = _get_headers(args)
    headers["HTTP-Referer"] = "http://devel.ops.knuth.li"
    headers["X-Title"] = "AI Chat"
    print(f"Model used is {state_payload['model']}")
    data = {
        "model": state_payload["model"],
        "messages": messages,
        "stream": True
    }

    full_text = ""
    sock = _open_socket(args) if STREAM_ENABLED else None
    _send(sock, {"state": state_payload})
    try:
        if STREAM_ENABLED:
            # Use proper SSE streaming implementation per OpenRouter docs
            with requests.post(url, headers=headers, json=data, stream=True) as response:
                print(f"Open API call ({url}, {headers}, {data}) sent, response in streamed")
                response.raise_for_status()
                
                # Set encoding explicitly to UTF-8
                response.encoding = 'utf-8'
                
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        # Handle Server-Sent Events format
                        if line.startswith('data: '):
                            data_str = line[6:]
                            if data_str == '[DONE]':
                                break
                            
                            try:
                                data_obj = json.loads(data_str)
                                if 'choices' in data_obj and len(data_obj['choices']) > 0:
                                    delta = data_obj['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        content = delta['content']
                                        full_text += content
                                        _send(sock, {"output": content})
                            except json.JSONDecodeError:
                                # Handle incomplete JSON chunks 
                                continue

            # Update history
            messages.append({"role": "assistant", "content": full_text})
            state_payload["history"] = messages
            
        else:
            # Non-streaming request
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            
            full_text = result['choices'][0]['message']['content']
            
            # Update history
            messages.append({"role": "assistant", "content": full_text})
            state_payload["history"] = messages

    except requests.exceptions.RequestException as e:
        full_text = f"API request failed: {str(e)}"
        if STREAM_ENABLED:
            _stream_text(sock, full_text)
    except Exception as e:
        full_text = f"An error occurred: {str(e)}"
        if STREAM_ENABLED:
            _stream_text(sock, full_text)
    finally:
        if STREAM_ENABLED:
            _close_socket(sock)

    return {"output": full_text, "streaming": STREAM_ENABLED, "state": state_payload}