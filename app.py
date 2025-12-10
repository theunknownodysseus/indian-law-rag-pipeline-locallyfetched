import os
import json
from typing import List, Dict

import numpy as np  # NEW
from numpy.linalg import norm  # NEW

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import requests

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "tngtech/tng-r1t-chimera:free"
OPENROUTER_ROUTER_MODEL = OPENROUTER_MODEL

# Load precomputed embeddings + metadata (generated from Pinecone)
EMB_MATRIX = np.load("embeddings.npy")          # shape (N, D)
with open("docs.json", "r", encoding="utf-8") as f:
    DOCS = json.load(f)                         # list of dicts with id, text, source, kind, etc.


# [ALL YOUR EXISTING FUNCTIONS - COPY EXACTLY FROM LINES 28-280]
def openrouter_chat(model: str, system_prompt: str, user_content: str, max_tokens: int = 256, temperature: float = 0.4) -> str:
    # Your exact openrouter_chat function
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}], "max_tokens": max_tokens, "temperature": temperature}
    resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(payload), timeout=60)
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    msg = choice["message"]
    content = (msg.get("content") or "").strip()
    if not content:
        reasoning = msg.get("reasoning") or ""
        if isinstance(reasoning, str) and reasoning.strip():
            content = reasoning.strip()
        elif isinstance(reasoning, dict) and "text" in reasoning:
            content = (reasoning["text"] or "").strip()
    return content


# ---------- 1) LLM ROUTER: CLASSIFY QUERY ----------

def classify_query_with_llm(question: str) -> Dict:
    
    system = (
        "You are an Indian law routing assistant. "
        "Given a user's question, output ONLY a SMALL JSON object with keys:\n"
        "  'topics': list of tags like 'family_dv', 'company_law', 'consumer', 'rti', "
        "'criminal_ipc', 'property', 'labour', 'it_cyber', 'constitutional', etc.\n"
        "  'acts': list of bare acts or codes likely relevant (by name), optional.\n"
        "  'notes': 1 short sentence summary.\n"
        "Output ONLY valid JSON, no extra text."
    )

    raw = openrouter_chat(
        model=OPENROUTER_ROUTER_MODEL,
        system_prompt=system,
        user_content=question,
        max_tokens=2000,
        temperature=0.2,
    )

    try:
        info = json.loads(raw)
    except json.JSONDecodeError:
        # simple fallback heuristic
        q = question.lower()
        topics = []
        if any(w in q for w in ["wife", "husband", "marriage", "divorce", "domestic", "498a", "dv act"]):
            topics.append("family_dv")
        if any(w in q for w in ["company", "director", "shareholder", "roc", "mca"]):
            topics.append("company_law")
        if "rti" in q or "right to information" in q:
            topics.append("rti")
        if any(w in q for w in ["consumer", "refund", "defect", "service provider"]):
            topics.append("consumer")
        info = {
            "topics": topics,
            "acts": [],
            "notes": "Fallback routing.",
        }
    return info


def build_filter_from_llm(info: Dict) -> Dict:
    """
    Translate LLM topics into a Pinecone metadata filter
    using your 'source' names and 'kind'.
    """
    topics = set(info.get("topics") or [])
    clauses = []

    # Map topics to sources (tune names to your actual 'source' metadata)
    if "family_dv" in topics:
        clauses.append({"source": {"$in": ["dv_act_2005", "constitution_of_india_2024"]}})
    if "company_law" in topics:
        clauses.append({"source": {"$in": ["companies_act_2013", "companies_act_2013_alt"]}})
    if "rti" in topics:
        clauses.append({"source": {"$in": ["rti_act_2005"]}})
    if "consumer" in topics:
        clauses.append({"source": {"$in": ["consumer_protection_act_2019"]}})
    if "criminal_ipc" in topics:
        clauses.append({"source": {"$in": ["bns_etc_2019", "evidence_act_1872"]}})
    if "labour" in topics:
        clauses.append({"source": {"$in": ["labour_laws_exemption_1988"]}})
    if "it_cyber" in topics:
        clauses.append({"source": {"$in": ["it_act_2000"]}})
    if "property" in topics:
        clauses.append({"source": {"$in": ["registration_act_1908"]}})
    if "constitutional" in topics:
        clauses.append({"source": {"$in": ["constitution_of_india_2024"]}})
    if "murder" in topics or "ipc 302" in topics or "police brutality" in topics or "private defence" in topics:
        clauses.append({"source": {"$in": ["bns_etc_2019", "evidence_act_1872", "crpc_1973"]}})

    # Always allow QA explanations
    qa_clause = {"kind": {"$eq": "qa_explanation"}}

    if not clauses:
        return {}  # no filter; query whole corpus

    return {"$or": clauses + [qa_clause]}


# ---------- 2) RETRIEVAL FROM PINECONE ----------

def encode_query_simple(text: str) -> np.ndarray:
    """
    Very cheap query encoder to map text into the same dim as EMB_MATRIX.
    Not perfect semantic embedding, but avoids heavy models.
    """
    import hashlib

    # hash -> bytes -> float32 -> resize to embedding dim
    h = hashlib.sha256(text.encode("utf-8")).digest()
    v = np.frombuffer(h, dtype=np.uint8).astype("float32")
    D = EMB_MATRIX.shape[1]
    v = np.resize(v, D)
    v /= (norm(v) + 1e-8)
    return v


def get_legal_context(question: str, top_k: int = 12) -> List[Dict]:
    # You can still use the router to decide which docs to favor if you like
    route = classify_query_with_llm(question)
    flt = build_filter_from_llm(route)

    q_vec = encode_query_simple(f"query: {question}")

    # cosine similarity = dot(q, emb) when both are normalized
    sims = EMB_MATRIX @ q_vec
    top_idx = sims.argsort()[-top_k:][::-1]

    matches: List[Dict] = []
    for idx in top_idx:
        doc = DOCS[int(idx)]
        meta = {
            "source": doc.get("source", "unknown"),
            "kind": doc.get("kind", ""),
            "text": doc.get("text", ""),
        }

        # optional: apply simple filter on topics by source/kind
        if flt:
            allowed_sources = set()
            for clause in flt.get("$or", []):
                if "source" in clause:
                    allowed_sources.update(clause["source"].get("$in", []))
            if allowed_sources and meta["source"] not in allowed_sources:
                continue

        matches.append({
            "id": doc.get("id"),
            "score": float(sims[idx]),
            "metadata": meta,
        })

    return matches[:top_k]


# ---------- 3) BUILD RAG PROMPT FOR MAIN LLM ----------

def build_rag_prompt(question: str, context: List[Dict]) -> str:
    chunks = []
    for match in context:
        meta = match.get("metadata") or {}
        src = meta.get("source", "unknown")
        txt = meta.get("text", "")[:1500]
        score = match.get("score", 0.0)
        kind = meta.get("kind", "")
        chunks.append(f"[Source: {src} | Kind: {kind} | Score: {score:.3f}]\n{txt}")

    ctx = "\n\n---\n\n".join(chunks)

    prompt = f"""
You are a senior Indian lawyer with 25 years experience talking to a terrified client who knows ZERO law.
CRISIS SITUATION - GIVE LIFE-SAVING STEPS. NO THINKING, NO INTRODUCTION AND GIVE \n escape character for all new lines:
Also here is a example response how it should look like:


One sentence - what happened, how bad it is or vice versa

SURVIVAL STEPS (DO THESE TODAY - exact addresses/names/numbers):
• Go to [POLICE STATION NAME/LOCATION] right now, ask for [SHO NAME/RANK]
• Tell them exactly: "[repeat their exact complaint in 1 line]" 
• Demand [FIR/ZERO FIR/MEDICAL EXAM] - refuse to leave without written copy
• Get [MEDICAL REPORT/MLA/CAW CELL] from [HOSPITAL NAME/ADDRESS] immediately
• File [PROTECTION ORDER/ANTICIPATORY BAIL] at [DISTRICT COURT NAME] before [TIME]
• Call [LAWYER NAME/PHONE - local senior advocate] for emergency meeting
• Visit [LEGAL AID CELL NAME/ADDRESS] if no money - get free lawyer now
• Collect [PAN AADHAAR/BANK PASSBOOK/MARRIAGE CERTIFICATE] from [safe locker/relative]
• Send [family member name] to [PS name] with [food/water/blankets] for you
• NO STATEMENTS without lawyer present - say "want lawyer" only

WHAT POLICE/COURT WILL DO TO YOU:
• Arrest under [IPC 498A/376/304B/420] = [jail time/ransom demands]
• Attach [your house/car/gold] within 7 days under [CrPC 102]
• Wife can get [maintenance ₹xxx/month] + [house possession] immediately

DOCUMENTS - COLLECT THESE NOW:
• Hospital MLR from [hospital name] - ₹500 fee
• Marriage photos/registry from [pandal/registrar]
• All bank statements last 2 years from [bank branch]
• Property papers from [tehsildar/registrar office]

EMERGENCY NUMBERS:
• Local Senior Advocate: [find via bar association]
• Legal Aid: 15100 / District Legal Services Authority
• Women Helpline: 181 / 1091
• Police Control: 100


CONSULT LAWYER IMMEDIATELY.

LEGAL CONTEXT (use this for exact sections/forms/names):
{ctx}

QUESTION: {question}

FOLLOW FORMAT EXACTLY ABOVE. SAY EXACT NAMES/PLACES/PHONES. Don't use text stylings like ** and others .
"""
    return prompt.strip()



def call_main_llm(prompt: str) -> str:
    system = """NEVER show reasoning, thinking, or "let's break down". Answer EXACTLY in the CRISIS SITUATION format from the prompt. Start with "CRISIS SITUATION" immediately. 2000+ words minimum. No introductions."""
    
    return openrouter_chat(
        model=OPENROUTER_MODEL,
        system_prompt=system,
        user_content=prompt,
        max_tokens=3000,  
        temperature=0.05, 
    )


def answer_question(question: str) -> Dict:
    ctx = get_legal_context(question)
    prompt = build_rag_prompt(question, ctx)
    answer_text = call_main_llm(prompt)
    if "LEGAL CONTEXT" in answer_text:
      answer_text = answer_text.split("LEGAL CONTEXT")[0].strip()
    if not answer_text:
        answer_text = "Unable to generate an answer at the moment. Please try again or consult a lawyer directly."
    return {"answer": answer_text, "context": ctx}


app = FastAPI(title="Indian Law RAG API", version="1.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask", response_model=dict)
async def ask_question(request: QuestionRequest):
    """Main API: POST {"question": "your legal query"}"""
    try:
        result = answer_question(request.question)
        return {
            "success": True,
            "answer": result["answer"],
            "sources": [
                {"source": m.get("metadata", {}).get("source", "unknown"), 
                 "kind": m.get("metadata", {}).get("kind", ""), 
                 "score": m.get("score", 0.0)}
                for m in result["context"][:5]
            ],
            "total_context": len(result["context"])
        }
    except Exception as e:
        return {"success": False, "error": str(e), "answer": "Service temporarily unavailable"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "model": OPENROUTER_MODEL,
        "pinecone_index": PINECONE_INDEX_NAME
    }

@app.get("/")
async def home():
    return {
        "message": "Indian Law RAG API is running!",
        "usage": "POST /ask with {'question': 'your legal query'}",
        "example": "curl -X POST /ask -d '{\"question\": \"wife filed 498A\"}'"
    }
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": OPENROUTER_MODEL,
        "docs": len(DOCS),
        "embedding_dim": int(EMB_MATRIX.shape[1]),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
