import os
import google.generativeai as genai

genai.configure(api_key=os.environ("AIzaSyBfI3lHrjVZE7KQajHUwhDawVJ7FuPRLrU"))

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="tunedModels/finalds-uq04usd3bkkz",
  generation_config=generation_config,
)

chat_session = model.start_chat(
  history=[
  ]
)

response = chat_session.send_message("INSERT_INPUT_HERE")

print(response.text)