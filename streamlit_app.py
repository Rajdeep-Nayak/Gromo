import streamlit as st
import os
import uuid
from googletrans import Translator
from gtts import gTTS
from langdetect import detect
from textblob import TextBlob
import google.generativeai as genai
from transformers import pipeline
from deep_translator import GoogleTranslator

# Streamlit page config
st.set_page_config(page_title="GroMo Assistant", layout="centered")
st.title("GromoGuide")

# --- API Keys ---
google_api_key = st.secrets["gcp"]["google_api_key"]
genai.configure(api_key=google_api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# --- Summarizer model (HuggingFace) ---
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# --- Utils ---
translator = Translator()

def translate(text, target_lang='en'):
    if target_lang not in ['hi', 'en']:
        return text
    return translator.translate(text, dest=target_lang).text

def generate_voice(text, filename=None):
    filename = filename or f"response_{uuid.uuid4().hex[:6]}.mp3"
    path = f"/tmp/{filename}"
    tts = gTTS(text=text, lang='hi')
    tts.save(path)
    return path

def clean(text): return text.strip().lower()

def detect_emotion(text):
    try:
        polarity = TextBlob(text).sentiment.polarity
        return "positive" if polarity > 0.3 else "negative" if polarity < -0.3 else "neutral"
    except:
        return "neutral"

tracked_keywords = ["credit", "loan", "upi", "demat", "savings", "investment", "home loan", "account"]
def extract_interests(text):
    return [kw for kw in tracked_keywords if kw in text.lower()]

def summarize_to_hindi(text):
    try:
        summary_en = summarizer(text, max_length=100, min_length=60, do_sample=False)[0]['summary_text']
        return GoogleTranslator(source='en', target='hi').translate(summary_en)
    except Exception:
        fallback_en = '. '.join(text.split('.')[:2])
        return GoogleTranslator(source='en', target='hi').translate(fallback_en)

# --- Knowledge Base (Abbreviated) ---
faq_knowledge = {
    # CREDIT CARD
    "what is credit card": "A credit card lets you buy things now and pay later. You get a limit based on your income and credit score.",
    "how to apply for credit card": "You need PAN card, Aadhaar card, income proof, and a mobile number. Application can be done online with GroMo.",
    "benefits of credit card": "You get cashback, reward points, free movie tickets, and can shop on EMI.",
    "how to sell credit card": "Ask customer what they want. Suggest card with rewards, low charges. Explain how to use it safely.",
    "credit card charges": "Joining fee, yearly fee, late payment fee. No interest if you pay full bill on time.",
    "credit card eligibility": "Age 18+, regular income, PAN and Aadhaar card needed.",
    "why credit card is useful": "You can shop, book tickets, and use in emergencies. Also helps build credit score.",
    "credit card se kaise fayda milega": "Sahi time par bill chukao, cashback aur rewards kamao. Credit score bhi sudharta hai.",

    # DEMAT ACCOUNT
    "what is demat account": "Demat account is for holding shares in digital form. It helps you invest in stock market.",
    "benefits of demat account": "Shares are safe, no paper needed, fast buying/selling of stocks.",
    "how to open demat account": "Need PAN, Aadhaar, photo, bank proof. Can be opened online in few minutes.",
    "how to sell demat account": "Tell customer they can invest in shares safely. Explain stock market returns.",
    "demat account ka use kya hai": "Share market me invest karne ke liye zaroori hai. Shares digital format me hote hain.",
    "demat vs savings account": "Demat holds shares. Savings holds money. Dono ka use alag hai.",

    # SAVINGS ACCOUNT
    "what is savings account": "A savings account is a bank account where you keep money safely and earn some interest.",
    "how to explain savings account": "Bachat account hai. Paise rakhne, ATM se nikalne aur UPI karne ke kaam aata hai.",
    "benefits of savings account": "Paise surakshit rehte hain, interest milta hai, ATM aur mobile banking milti hai.",
    "documents for savings account": "Aadhaar card, PAN card, photo, and phone number.",
    "how to sell savings account": "Daily use ke liye safe account hai. UPI, ATM, online banking sab milega.",

    # LOAN
    "what is loan": "Loan means borrowing money from bank which you return in parts with interest.",
    "types of loans": "Personal loan, home loan, car loan, education loan etc.",
    "how to pitch a loan": "Pehle samjho customer ko paise kis kaam ke liye chahiye. Fir loan ka amount, interest aur documents batao.",
    "loan ke liye kya chahiye": "PAN, Aadhaar, income proof, photo, aur account details chahiye.",
    "benefits of loan": "Zarurat ke waqt bada paisa milta hai. Ghar, car, education ke liye useful hai.",
    "loan objections": "Interest zyada lagta hai, documents ka issue. Clear karo ki GroMo ke through asaan process hai.",

    # INVESTMENT
    "what is investment": "Investment means putting money in things like mutual funds, stocks, or gold to earn more money over time.",
    "benefits of investment": "Aapka paisa badhta hai. Savings se zyada return milta hai.",
    "types of investment": "Mutual funds, SIP, shares, gold, FD, etc.",
    "how to explain investment": "Simple language me samjhao ki paisa bank me sirf rehta hai, invest karne se badhta hai.",
    "how to sell investment": "Pehle poochho risk lena chahte hain ya safe option chahiye. Fir mutual fund ya SIP suggest karo.",
    "documents for investment": "PAN, Aadhaar, bank account, photo chahiye.",

    # CREDIT LINE
    "what is credit line": "Credit line gives you a limit, like a loan, which you can use anytime and repay slowly.",
    "credit line vs loan": "Loan ek baar milta hai. Credit line bar bar use kar sakte ho limit ke andar.",
    "how to sell credit line": "Batayein ki ye flexible loan jaisa hai. Business ya personal zarurat me kaam aata hai.",
    "benefits of credit line": "Emergency me paisa turant milta hai. Sirf jitna use karo utna hi interest lagta hai.",
    "eligibility for credit line": "Income proof, PAN, Aadhaar, aur account details chahiye.",

    # UPI
    "what is upi": "UPI ek payment method hai jisse aap mobile se turant paisa bhej ya le sakte ho.",
    "benefits of upi": "Instant transfer, no charges, 24x7 available, safe and fast.",
    "how to use upi": "Google Pay, PhonePe, Paytm jaisi apps me UPI ID banao aur bank account link karo.",
    "upi se kaise paisa bheje": "App open karo, UPI ID ya mobile number daalo, amount likho aur PIN se bhejo.",
    "how to sell upi account": "Batayein ki digital payment karna easy hai, cash carry karne ki zarurat nahi.",

    # SALES & LEAD CONVERSION
    "how to convert lead": "Follow-up karo, customer ka doubt clear karo, unki need samjho aur sahi product suggest karo.",
    "how to sell financial products": "Har product ka simple benefit batao, examples do, aur paperwork easy hone ka assurance do.",
    "how to build trust with customer": "Seedha aur sach bolo. Unki baat dhyan se suno. Koi bhi doubt ho to clear karo.",
    "how to handle objections": "Customer ka concern calmly suno. Right info do. Misunderstanding clear karo.",
    "what to do if lead is not responding": "Friendly reminder bhejo, call karo. Value batao. Time do. Push na karo.",
    "how to follow up": "WhatsApp ya call se politely follow-up karo. Remind karo ki product unke kaam ka hai.",

    # ONBOARDING & GENERAL
    "how does gromo help": "GroMo app se aap financial products bech sakte ho. Training, support aur commission sab milta hai.",
    "what is gromo": "GroMo ek platform hai jaha aap bank products jaise loan, credit card, account bechkar paise kama sakte ho.",
    "documents needed to start": "Aapko Aadhaar, PAN aur bank account chahiye GroMo par kaam shuru karne ke liye.",
    "how to earn with gromo": "Products bechne par aapko har sale par commission milta hai. Jitna bechoge utna kamaoge."
}
# --- Chat Logic ---
def chatbot_response(user_input):
    try:
        detected_lang = detect(user_input)
        query_in_english = translate(user_input, target_lang='en')
        emotion = detect_emotion(query_in_english)
        interests = extract_interests(query_in_english)

        cleaned_query = clean(query_in_english)
        if cleaned_query in faq_knowledge:
            response_en = faq_knowledge[cleaned_query]
        else:
            system_prompt = """
You are a smart and helpful assistant built for GroMo Partners.
Your job is to answer queries about 8 financial products:
Credit Card, Home Loan, Demat Account, Savings Account, Loan, Investment, Credit Line, UPI.

You also help agents learn how to:
- Sell these products effectively
- Handle customer objections
- Pitch products clearly and confidently
- Convert leads using best practices

Be simple, specific, and supportive.
"""
            prompt = f"{system_prompt}\n\nUser: {query_in_english}\n\nAssistant:"
            gemini_response = model.generate_content(prompt)
            response_en = gemini_response.text.strip()

        response_hi = summarize_to_hindi(response_en)
        audio_path = generate_voice(response_hi)

        return {
            "english": response_en,
            "hindi": response_hi,
            "audio_path": audio_path,
            "interests": interests
        }

    except Exception as e:
        return {"error": str(e)}

# --- Streamlit UI ---
st.header("💬 Ask Your Questions")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_question" not in st.session_state:
    st.session_state.last_question = ""

user_input = st.text_input("Type your question and press Enter")

if user_input:
    full_input = f"{st.session_state.last_question}\nUser: {user_input}"
    result = chatbot_response(full_input)

    if "error" in result:
        st.error(result["error"])
    else:
        st.session_state.chat_history.append({
            "question": user_input,
            "english": result["english"],
            "hindi": result["hindi"],
            "audio_path": result["audio_path"],
            "interests": result["interests"]
        })
        st.session_state.last_question = user_input

# --- Chat Display ---
for entry in reversed(st.session_state.chat_history):
    st.markdown(f"**🧑 You:** {entry['question']}")
    st.markdown(f"**🔤 English:** {entry['english']}")
    st.markdown(f"**🗣 Hindi:** {entry['hindi']}")
    if entry["interests"]:
        st.markdown(f"**🔎 Interests Detected:** {', '.join(entry['interests'])}")
    if entry["audio_path"]:
        st.audio(entry["audio_path"], format="audio/mp3")
    st.markdown("---")
