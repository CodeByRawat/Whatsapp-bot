import os
import time
import threading
from collections import defaultdict, deque

import pandas as pd
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
from twilio.rest import Client
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_community.llms import LlamaCpp

# ----------------- ENV -----------------
load_dotenv()

ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
FROM_WHATSAPP = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_PATH = os.getenv("MODEL_PATH")

TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "180"))
N_CTX = int(os.getenv("LLM_N_CTX", "2048"))
N_THREADS = int(os.getenv("LLM_N_THREADS", "4"))

EXCEL_PATH = os.path.abspath(os.getenv("EXCEL_PATH", "contacts.xlsx"))
DEFAULT_CC = os.getenv("DEFAULT_COUNTRY_CODE", "+91")

AUTO_MESSAGE = (os.getenv("AUTO_MESSAGE") or "Hi").strip()

if not ACCOUNT_SID or not AUTH_TOKEN:
    raise RuntimeError("Missing Twilio credentials in .env")

twilio_client = Client(ACCOUNT_SID, AUTH_TOKEN)
app = Flask(__name__)

# ----------------- HELPERS -----------------
def normalize_e164(raw: str) -> str:
    if raw.startswith("+"):
        return raw
    digits = "".join(ch for ch in raw if ch.isdigit())
    cc = DEFAULT_CC if DEFAULT_CC.startswith("+") else f"+{DEFAULT_CC}"
    return f"{cc}{digits}"

def load_profiles(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    df = pd.read_excel(path)
    if not {"Name", "Phone"} <= set(df.columns):
        return {}
    if "Context" not in df.columns:
        df["Context"] = ""
    return {
        normalize_e164(str(r["Phone"])): {
            "name": str(r["Name"]).strip(),
            "context": str(r["Context"]).strip()
        }
        for _, r in df.iterrows()
    }

PROFILES = load_profiles(EXCEL_PATH)

# ----------------- LLM -----------------
def build_llm():
    if OPENAI_API_KEY:
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=TEMPERATURE)
    if MODEL_PATH:
        return LlamaCpp(
            model_path=MODEL_PATH,
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            n_batch=32,
            n_gpu_layers=0,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
    raise RuntimeError("No LLM configured")

LLM = build_llm()
PROMPT = PromptTemplate(
    input_variables=["history", "message", "profile"],
    template=(
        "You are a friendly WhatsApp assistant. Keep replies short.\n"
        "User profile: {profile}\n"
        "Conversation:\n{history}\n"
        "User: {message}\nAssistant:"
    )
)
CHAIN = LLMChain(llm=LLM, prompt=PROMPT)
HISTORY = defaultdict(lambda: deque(maxlen=12))

def get_profile(sender_whatsapp: str) -> str:
    e164 = sender_whatsapp.replace("whatsapp:", "").strip()
    prof = PROFILES.get(e164)
    if not prof:
        return ""
    name, ctx = prof.get("name", ""), prof.get("context", "")
    return f"Name: {name}. {ctx}".strip()

# ----------------- BROADCAST -----------------
def broadcast_message(text: str):
    results = []
    for e164 in PROFILES:
        try:
            msg = twilio_client.messages.create(
                from_=FROM_WHATSAPP,
                to=f"whatsapp:{e164}",
                body=text
            )
            results.append({"to": e164, "sid": msg.sid, "status": "sent"})
            time.sleep(0.3)
        except Exception as e:
            results.append({"to": e164, "error": str(e), "status": "failed"})
    return results

def auto_broadcast_on_start():
    if PROFILES:
        broadcast_message(AUTO_MESSAGE)

# ----------------- ROUTES -----------------
@app.get("/health")
def health():
    return {"ok": True, "profiles": len(PROFILES)}

@app.post("/reload-profiles")
def reload_profiles():
    global PROFILES
    PROFILES = load_profiles(EXCEL_PATH)
    return {"ok": True, "profiles": len(PROFILES)}

@app.post("/broadcast")
def broadcast_message(text: str):
    results = []
    print(f"[broadcast] Sending to {len(PROFILES)} contact(s): {text!r}")
    for e164 in PROFILES:
        try:
            msg = twilio_client.messages.create(
                from_=FROM_WHATSAPP,
                to=f"whatsapp:{e164}",
                body=text
            )
            print(f"[sent] {e164} SID={msg.sid}")
            results.append({"to": e164, "sid": msg.sid, "status": "sent"})
            time.sleep(0.3)
        except Exception as e:
            print(f"[failed] {e164}: {e}")
            results.append({"to": e164, "error": str(e), "status": "failed"})
    print(f"[broadcast done] sent={sum(1 for r in results if r['status']=='sent')} "
          f"failed={sum(1 for r in results if r['status']=='failed')}")
    return results

@app.post("/webhook")
def webhook():
    sender = request.values.get("From", "").strip()
    body = request.values.get("Body", "").strip()
    if not sender or not body:
        return Response("ok", status=200)

    HISTORY[sender].append(f"User: {body}")
    profile = get_profile(sender)
    history_str = "\n".join(HISTORY[sender])

    def worker():
        reply = (PROMPT | LLM).invoke({
            "history": history_str,
            "message": body,
            "profile": profile
        }).strip() or "Sorry, I’m having trouble replying."
        HISTORY[sender].append(f"Assistant: {reply}")
        twilio_client.messages.create(from_=FROM_WHATSAPP, to=sender, body=reply)

    threading.Thread(target=worker, daemon=True).start()
    return Response("ok", status=200)

# ----------------- MAIN -----------------
if __name__ == "__main__":
    auto_broadcast_on_start()
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)

