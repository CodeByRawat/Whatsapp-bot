WhatsApp Chatbot with Local LLM (Mistral/Qwen) + Twilio

This is a Flask-based WhatsApp chatbot that:

    Sends an auto “Hi” message to all contacts in an Excel file on startup.

    Uses local LlamaCpp (Mistral/Qwen/etc. in GGUF format) to generate replies.

    Receives messages via Twilio WhatsApp Webhook and replies in real time.

    Logs INCOMING and OUTGOING messages in the terminal.

--> Features

    Local LLM inference 

    Auto-message on startup from Excel contact list.

    Incoming/Outgoing message logs.

    Configurable via .env file.

2️⃣ Requirements

    Python 3.10+

    A Twilio account with WhatsApp sandbox enabled.

    Local GGUF model file (e.g., Mistral-7B-Instruct Q4_K_M).

    Excel contact list.

3️⃣ Installation

# 1. Clone your project folder
git clone <your-repo-url>
cd <your-project-folder>

# 2. Create a Python virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install flask python-dotenv twilio pandas openpyxl langchain langchain-community llama-cpp-python

flask
python-dotenv
twilio
pandas
openpyxl
langchain
langchain-community
llama-cpp-python

4️⃣ Setup .env File

Create a .env file in the project root:

# Twilio Credentials
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886

# Local model path (GGUF)
MODEL_PATH=your_path/mistral-7b-instruct-v0.2.Q4_K_M.gguf

# LLM Parameters
LLM_N_THREADS=4
LLM_N_CTX=1024
LLM_MAX_TOKENS=60
LLM_TEMPERATURE=0.5

# Excel path + country code
EXCEL_PATH=excel_path
DEFAULT_COUNTRY_CODE=+91

5️⃣ Excel Format

contacts.xlsx should contain:
Name	Phone	Context (optional)
Alice	+91234567891 Frequent customer
Bob	+91234567893	

6️⃣ Running the Bot

python app.py

    On startup, it will send Hi to all contacts.

    Flask will run on port 5000.

    Connect it to Twilio via ngrok:

ngrok http 5000

Set the ngrok URL in Twilio Sandbox Inbound Webhook.

7️⃣ Endpoints

    GET /health → Check status of bot, model, and contacts.

    POST /reload-profiles → Reload contacts.xlsx without restarting server.

    POST /test-send → Send a manual test message.

    POST /webhook → Twilio inbound WhatsApp messages.

8️⃣ Logs

Example terminal logs:

[AUTO-SEND] To: whatsapp:+91234564789 | SID: SMxxx | Status: sent
[INCOMING] From: whatsapp:+91234564789 | Message: Hi
[OUTGOING] To: whatsapp:+91234564789 | SID: SMxxx | Body: Hello! How can I help you today?

