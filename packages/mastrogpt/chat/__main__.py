#--kind python:default
#--web true
#--param OLLAMA_API_HOST "$OLLAMA_API_HOST"
#--param OLLAMA_API_SECRET "$OLLAMA_API_SECRET"

import chat
def main(args):
  return { "body": chat.chat(args) }
