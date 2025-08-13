# app.py — WhatsApp bot (Twilio + Flask) with local LlamaCpp (Mistral/Qwen/etc), auto-“Hi” on startup,
#           and simple conversation replies + clear IN/OUT logging.
#
# What it does
# 1) On server start, reads contacts.xlsx and sends "Hi" to every number (one-by-one, with pacing).
# 2) When someone replies on WhatsApp, generates a short reply using your LOCAL GGUF model
#    (via llama-cpp-python) in a background thread, then sends it via Twilio REST.
# 3) Logs every incoming and outgoing message in the terminal:
#       [AUTO-SEND] / [INCOMING] / [OUTGOING]
#
# Requirements (install in your env):
#   pip install flask python-dotenv twilio pandas openpyxl langchain langchain-community llama-cpp-python
#
# .env (put next to this file). Example:
#   TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxx     # or use API key auth (see below)
#   TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#   # Accept either of these var names for the sender:
#   TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
#   # or
#   # TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
#
#   # LOCAL model (GGUF path)
#   MODEL_PATH=S:/Whatsapp chatbot project/Models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
#   # Suggested smaller models for speed on CPU:
#   # MODEL_PATH=S:/Whatsapp chatbot project/Models/qwen2-1_5b-instruct.Q4_K_M.gguf
#   # MODEL_PATH=S:/Whatsapp chatbot project/Models/mistral-nemo-2b-instruct.Q4_K_M.gguf
#
#   # Llama.cpp tuning
#   LLM_N_THREADS=4
#   LLM_N_CTX=1024
#   LLM_MAX_TOKENS=60
#   LLM_TEMPERATURE=0.5
#
#   # Excel + defaults
#   EXCEL_PATH=S:/Whatsapp chatbot project/contacts.xlsx
#   DEFAULT_COUNTRY_CODE=+91
#
# Notes:
# - Twilio Sandbox can ONLY message numbers that have joined your sandbox once.
# - Auto-send on startup will consume your daily Twilio quota. Disable it by commenting out send_hi_to_all()
#   in __main__ if you hit daily limit errors (e.g., 63038).

import os
import time
import threading
import traceback
from collections import defaultdict, deque

import pandas as pd
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
from twilio.rest import Client

from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp

# ------------------- Load env -------------------
load_dotenv()

ACCOUNT_SID   = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
AUTH_TOKEN    = os.getenv("TWILIO_AUTH_TOKEN", "").strip()

# Accept either var name for the sender to avoid confusion
FROM_WHATSAPP = (
    os.getenv("TWILIO_WHATSAPP_NUMBER")
    or os.getenv("TWILIO_WHATSAPP_FROM")
    or "whatsapp:+14155238886"
).strip()

MODEL_PATH    = os.getenv("MODEL_PATH", "").strip()

# LLM knobs for CPU
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.5"))
MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "60"))
N_CTX       = int(os.getenv("LLM_N_CTX", "1024"))
N_THREADS   = int(os.getenv("LLM_N_THREADS", "4"))

# Contacts
EXCEL_PATH  = os.path.abspath(os.getenv("EXCEL_PATH", "contacts.xlsx"))
DEFAULT_CC  = os.getenv("DEFAULT_COUNTRY_CODE", "+91").strip()

# ---- Basic sanity (no strict AC/SK checks to keep flexible) ----
if not ACCOUNT_SID or not AUTH_TOKEN:
    raise RuntimeError("Missing Twilio credentials in .env (TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN)")
if not FROM_WHATSAPP.startswith("whatsapp:"):
    print("⚠️  Warning: FROM_WHATSAPP is not in 'whatsapp:+<E164>' format. Value =", FROM_WHATSAPP)
if not MODEL_PATH:
    raise RuntimeError("Set MODEL_PATH in .env to your GGUF model path")

# Twilio client
# If you prefer API Key auth instead of auth token:
#  - set TWILIO_API_KEY_SID and TWILIO_API_KEY_SECRET in .env and use:
#      Client(username=API_KEY_SID, password=API_KEY_SECRET, account_sid=ACCOUNT_SID)
API_KEY_SID    = os.getenv("TWILIO_API_KEY_SID", "").strip()
API_KEY_SECRET = os.getenv("TWILIO_API_KEY_SECRET", "").strip()
if API_KEY_SID and API_KEY_SECRET:
    twilio_client = Client(username=API_KEY_SID, password=API_KEY_SECRET, account_sid=ACCOUNT_SID)
else:
    twilio_client = Client(ACCOUNT_SID, AUTH_TOKEN)

app = Flask(__name__)

# ------------------- Contacts helpers -------------------
def normalize_e164(raw: str) -> str:
    """Convert '8004466229' → '+918004466229' using DEFAULT_CC; keep '+..' as-is."""
    s = str(raw).strip()
    if s.startswith("+"):
        return s
    digits = "".join(ch for ch in s if ch.isdigit())
    cc = DEFAULT_CC if DEFAULT_CC.startswith("+") else f"+{DEFAULT_CC}"
    return f"{cc}{digits}"

def load_profiles(path: str) -> dict:
    """
    Excel columns (case-sensitive):
      - Name   (required)
      - Phone  (required)
      - Context (optional)
    Returns dict: '+91XXXXXXXXXX' → {"name":..., "context":...}
    """
    if not os.path.exists(path):
        print(f"[warn] Excel not found: {path}")
        return {}
    df = pd.read_excel(path)
    need = {"Name", "Phone"}
    missing = need - set(df.columns)
    if missing:
        print(f"[warn] Excel missing columns {missing}")
        return {}
    if "Context" not in df.columns:
        df["Context"] = ""
    profiles = {}
    for _, r in df.iterrows():
        profiles[normalize_e164(r["Phone"])] = {
            "name": str(r["Name"]).strip(),
            "context": str(r.get("Context", "")).strip()
        }
    return profiles

PROFILES = load_profiles(EXCEL_PATH)

def get_profile(sender_whatsapp: str) -> str:
    """Map 'whatsapp:+91xxxxxxxxxx' → '+91xxxxxxxxxx' and fetch compact profile string."""
    e164 = sender_whatsapp.replace("whatsapp:", "").strip()
    prof = PROFILES.get(e164)
    if not prof:
        return ""
    bits = []
    if prof.get("name"):
        bits.append(f"Name: {prof['name']}")
    if prof.get("context"):
        bits.append(prof["context"])
    return ". ".join(bits)

# ------------------- LLM (local only) -------------------
def build_llm():
    print(f"LLM: using local LlamaCpp -> {MODEL_PATH}")
    return LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,            # 768–1024 is fine for WhatsApp chats
        n_threads=N_THREADS,    # 4 on your CPU
        n_batch=32,             # helps CPU throughput
        n_gpu_layers=0,         # CPU-only
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,  # 40–80 recommended to avoid long gens
        verbose=False,
    )

LLM = build_llm()

PROMPT = PromptTemplate(
    input_variables=["history", "message", "profile"],
    template=(
        "You are a concise WhatsApp assistant. Reply in <=2 short, helpful sentences.\n"
        "Profile: {profile}\n\n"
        "{history}\nUser: {message}\nAssistant:"
    )
)

# Runnable chain (Prompt → LLM) without deprecated LLMChain
CHAIN = PROMPT | LLM

# Simple rolling memory
HISTORY = defaultdict(lambda: deque(maxlen=12))
def format_history(sender_key: str) -> str:
    return "\n".join(HISTORY[sender_key])

# ------------------- Auto-send “Hi” on startup -------------------
def send_hi_to_all(message_text: str = "Hi"):
    if not PROFILES:
        print("[AUTO-SEND] No profiles loaded; skipping.")
        return
    print(f"[AUTO-SEND] Sending '{message_text}' to {len(PROFILES)} contact(s)...")
    ok = fail = 0
    for e164 in PROFILES:
        try:
            m = twilio_client.messages.create(
                from_=FROM_WHATSAPP,
                to=f"whatsapp:{e164}",
                body=message_text
            )
            print(f"[AUTO-SEND] To: whatsapp:{e164} | SID: {m.sid} | Status: {m.status}")
            ok += 1
            time.sleep(0.35)  # gentle pacing for sandbox
        except Exception as e:
            print(f"[AUTO-SEND] FAIL -> {e164}: {e}")
            fail += 1
    print(f"[AUTO-SEND] Done. sent={ok}, failed={fail}, total={len(PROFILES)}")

# ------------------- Routes -------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "profiles": len(PROFILES),
        "from": FROM_WHATSAPP,
        "model_path": MODEL_PATH,
        "excel_exists": os.path.exists(EXCEL_PATH),
    }

@app.post("/reload-profiles")
def reload_profiles():
    global PROFILES
    PROFILES = load_profiles(EXCEL_PATH)
    return {"ok": True, "profiles": len(PROFILES)}

@app.post("/test-send")
def test_send():
    """POST JSON: { "to": "+91XXXXXXXXXX", "text": "Hello!" }"""
    data = request.get_json(silent=True) or {}
    to = (data.get("to") or "").strip()
    text = (data.get("text") or "Test from server").strip()
    if not to.startswith("+"):
        return {"ok": False, "error": "Use +E.164 like +91xxxxxxxxxx"}, 400
    try:
        m = twilio_client.messages.create(from_=FROM_WHATSAPP, to=f"whatsapp:{to}", body=text)
        print(f"[OUTGOING] To: whatsapp:{to} | SID: {m.sid} | Body: {text}")
        return {"ok": True, "sid": m.sid}
    except Exception as e:
        return {"ok": False, "error": str(e)}, 400

@app.post("/webhook")
def webhook():
    """
    WhatsApp inbound → background-generate reply → send via REST.
    Twilio posts x-www-form-urlencoded (keys: Body, From, etc.).
    """
    try:
        sender = (request.values.get("From") or "").strip()   # 'whatsapp:+91...'
        body   = (request.values.get("Body") or "").strip()
        if not sender or not body:
            return Response("ok", status=200)

        print(f"[INCOMING] From: {sender} | Message: {body}")

        # Update memory now
        HISTORY[sender].append(f"User: {body}")
        profile = get_profile(sender)
        hist_str = format_history(sender)

        def worker():
            try:
                out = CHAIN.invoke({"history": hist_str, "message": body, "profile": profile})
                reply = out if isinstance(out, str) else str(out)
                reply = reply.strip() or "Sorry, I’m having trouble replying right now."

                HISTORY[sender].append(f"Assistant: {reply}")

                m = twilio_client.messages.create(from_=FROM_WHATSAPP, to=sender, body=reply)
                print(f"[OUTGOING] To: {sender} | SID: {m.sid} | Body: {reply}")
            except Exception as e:
                print("[bg error]", e)
                traceback.print_exc()

        threading.Thread(target=worker, daemon=True).start()
        return Response("ok", status=200)

    except Exception as e:
        print("[webhook error]", e)
        traceback.print_exc()
        return Response("ok", status=200)

# ------------------- Main -------------------
if __name__ == "__main__":
    print("Twilio from:", FROM_WHATSAPP)
    print("Excel path:", EXCEL_PATH)
    print("Profiles loaded:", len(PROFILES))

    # 1) Auto-send “Hi” to everyone on startup (comment out if you want to save daily quota)
    send_hi_to_all("Hi")

    # 2) Start the server
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)
