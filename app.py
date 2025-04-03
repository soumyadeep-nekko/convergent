from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import requests, json, os, datetime
import fitz

app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)

# --- Load secrets ---
def load_dict_from_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

secrets_file = "../secrets.json"
SECRETS = load_dict_from_json(secrets_file)

# ========== CONFIGURATION ==========
aws_access_key_id = SECRETS["aws_access_key_id"]
aws_secret_access_key = SECRETS["aws_secret_access_key"]
INFERENCE_PROFILE_ARN = SECRETS["INFERENCE_PROFILE_ARN"]
REGION = SECRETS["REGION"]

# --- PDF Text Extraction using PyMuPDF ---
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    extracted_text = "\n".join([page.get_text("text") for page in doc])
    return extracted_text

company_info_text = extract_text_from_pdf("document.pdf")

# --- LLM Call Function ---
def call_llm_api(conversation_history):
    url = AZURE_OPENAI_URL
    headers = {  
        "Content-Type": "application/json",  
        "api-key": AZURE_OPENAI_KEY  
    }  
    system_message = f"""
    You are Nekko Website Chatbot. Below is the company information and product details:
    {company_info_text}
    Nekko is an Industry Leader in AI ML Solutions.
    
    Your job is to:
    0. Collect Customer Details Such as Name and Mobile Number Mandatorily and if possible other information such as Email, Organisation Name as well.
    1. Present the latest and exciting product offerings by Nekko.
    2. Be empathetic and address the customer's pain points.
    3. Respond normally if the query is unrelated.
    4. Provide 'prithvi@nekko.tech' if the user asks for the sales team contact.
    5. Engage in a normal conversation.

    Since this is for a chatbot, Please avoid using markdown formatting. Keep your answers short and suitable for a chat environment.
    """
    messages = [{"role": "system", "content": system_message}] + conversation_history
    payload = {  
        "messages": messages,  
        "temperature": 0.7,  
        "max_tokens": 512   
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    response.raise_for_status()  
    return response.json()["choices"][0]["message"]["content"]

# --- Conversation File Handling ---
def get_conversation_file():
    # Returns the file path for the current session's conversation.
    if "conversation_file" in session:
        return session["conversation_file"]
    conv_file = os.path.join("conversations", f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    session["conversation_file"] = conv_file
    return conv_file

# Ensure conversations folder exists.
if not os.path.exists("conversations"):
    os.makedirs("conversations")

@app.route('/')
def index():
    # Initialize session variables for new conversation.
    session['chat_history'] = []
    get_conversation_file()  # Sets the conversation file path.
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_query = data.get("user_query")
    if not user_query:
        return jsonify({"error": "No user query provided"}), 400

    # Retrieve and update conversation history from session.
    chat_history = session.get('chat_history', [])
    chat_history.append({"role": "user", "content": user_query})

    # Call the LLM to generate a reply.
    trimmed_history = chat_history[-10:]  # Using the last 10 messages.
    try:
        reply = call_llm_api(trimmed_history)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    chat_history.append({"role": "assistant", "content": reply})
    session['chat_history'] = chat_history

    # Save the updated conversation to a file.
    conv_file = get_conversation_file()
    print("Conversation file will be saved at:", os.path.abspath(conv_file))
    with open(conv_file, "w") as f:
        json.dump(chat_history, f, indent=4)

    return jsonify({"reply": reply})

if __name__ == '__main__':
    app.run(debug=True)
