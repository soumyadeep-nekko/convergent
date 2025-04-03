import os, json, time, re, datetime, requests

# --- Load secrets ---
def load_dict_from_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

secrets_file = "../secrets.json"
SECRETS = load_dict_from_json(secrets_file)

aws_access_key_id = SECRETS["aws_access_key_id"]
aws_secret_access_key = SECRETS["aws_secret_access_key"]
INFERENCE_PROFILE_ARN = SECRETS["INFERENCE_PROFILE_ARN"]
REGION = SECRETS["REGION"]

# --- LLM Call for Lead Extraction ---
def extract_lead_details_from_conversation(conversation):
    extraction_prompt = """
    You are tasked with extracting the following details from this conversation:
    - Name (if provided)
    - Phone Number
    - Email
    - Any pain points or comments shared by the user
    
    Return the information as a JSON object in the following format:
    ```json
        {
        "name": "",
        "phone": "",
        "email": "",
        "pain_points": ""
        }
    ```
    """
    messages = [{"role": "system", "content": extraction_prompt}]
    # Add the entire conversation
    for msg in conversation:
        messages.append(msg)
    payload = {"messages": messages, "temperature": 0.2, "max_tokens": 300}
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_KEY
    }
    response = requests.post(AZURE_OPENAI_URL, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    answer = response.json()["choices"][0]["message"]["content"]
    try:
        # return json.loads(answer[7:-3])
        return json.loads(answer.split("```json")[1].split("```")[0])
    except:
        # return json.loads(answer[3:-3])
        return json.loads(answer.split("```")[1].split("```")[0])

# --- File paths and folders ---
CONV_FOLDER = "conversations"
CONTACTS_FOLDER = "contacts"

if not os.path.exists(CONTACTS_FOLDER):
    os.makedirs(CONTACTS_FOLDER)

# --- Process Conversation Files ---
processed_files = set()

while True:
    files = os.listdir(CONV_FOLDER)
    for file in files:
        if file.endswith(".json") and file not in processed_files:
            filepath = os.path.join(CONV_FOLDER, file)
            try:
                with open(filepath, "r") as f:
                    conversation = json.load(f)
                # Extract lead details using the LLM
                lead_data = extract_lead_details_from_conversation(conversation)
                # Optionally, add a check to save only if mandatory fields are present.
                lead = lead_data
                if lead.get("name") and lead.get("phone"):
                    contact_file = os.path.join(CONTACTS_FOLDER, f"lead_{file}")
                    with open(contact_file, "w") as cf:
                        json.dump(lead, cf, indent=4)
                    print(f"[{datetime.datetime.now()}] Extracted and saved lead from {file} to {contact_file}")
                else:
                    print(f"[{datetime.datetime.now()}] Lead details not complete in {file}.")
                processed_files.add(file)
            except Exception as e:
                print(f"Error processing {file}: {e}")
    # Wait for a while before checking again (e.g., 10 seconds)
    time.sleep(10)
