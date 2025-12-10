# Indian Law RAG Pipeline

> A lightweight, crisis-focused legal Q&A system for Indian law that actually fits in a free-tier deployment.

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Railway](https://img.shields.io/badge/Railway-deployed-success)](https://railway.app/)

## What This Is

This is a Retrieval-Augmented Generation (RAG) pipeline that answers questions about Indian law. Ask it anything from "how do I file for divorce" to "what are my rights under consumer protection law," and it retrieves relevant sections from a corpus of major Indian statutes, then generates practical, step-by-step guidance.

The system covers central legislation including IPC, CrPC, CPC, Evidence Act, Constitution of India, Consumer Protection Act, Contract Act, IT Act, Motor Vehicles Act, Companies Act, GST, Domestic Violence Act, POSH, RTI, Juvenile Justice Act, and key Environment Acts.

**Live API:** `https://indian-law-rag-pipeline-production.up.railway.app`

---

## Why This Exists

### The Problem

Legal information is dense, scattered across hundreds of acts, and often inaccessible to people who need it most. When someone faces an urgent legal situation—domestic violence, workplace harassment, consumer fraud—they don't have time to read through entire statutes or afford immediate legal consultation.

Traditional legal chatbots either:
- Hallucinate case law and statutory provisions
- Require massive infrastructure (vector databases, GPU instances)
- Can't deploy on free-tier services because of bloated dependencies

### The Solution

This project takes a different approach:

**1. Retrieval-First Architecture**  
Instead of relying purely on LLM memory, we embed and index the actual text of Indian laws. When you ask a question, the system retrieves the exact sections that apply, then uses those as context for generation. This grounds answers in real statutory language.

**2. Offline Embedding Pipeline**  
The heavy lifting—parsing PDFs, generating embeddings, building the search index—happens once, offline. The production app loads precomputed `embeddings.npy` and `docs.json` files (around 50MB total) instead of shipping PyTorch and massive ML libraries. This keeps the Docker image under Railway's 4GB limit.

**3. Crisis-Format Responses**  
Answers aren't academic legal analysis. They're formatted as survival instructions: immediate steps, who to contact, what evidence to gather, timelines that matter. The system treats every query as potentially urgent.

---

## How It Works

### Architecture Overview

```
┌─────────────┐
│   User      │
│  Question   │
└──────┬──────┘
       │
       v
┌─────────────────────────────────────────────┐
│  1. Router LLM (Chimera)                    │
│     Classifies question into topics:        │
│     family, criminal, consumer, IT, etc.    │
└──────┬──────────────────────────────────────┘
       │
       v
┌─────────────────────────────────────────────┐
│  2. Embedding Search                        │
│     • Encode question → vector              │
│     • Cosine similarity vs embeddings.npy   │
│     • Filter by topic tags from router      │
│     • Return top-k chunks                   │
└──────┬──────────────────────────────────────┘
       │
       v
┌─────────────────────────────────────────────┐
│  3. RAG Prompt Construction                 │
│     • Inject retrieved law sections         │
│     • Add crisis-format instructions        │
│     • Include user's original question      │
└──────┬──────────────────────────────────────┘
       │
       v
┌─────────────────────────────────────────────┐
│  4. Generation (Chimera via OpenRouter)     │
│     Produces step-by-step legal guidance    │
└──────┬──────────────────────────────────────┘
       │
       v
┌─────────────────────────────────────────────┐
│  Response: answer + sources + metadata      │
└─────────────────────────────────────────────┘
```

### Offline Pipeline (Not Shipped)

The data preparation step runs locally or in a separate build:

1. **PDF Parsing**: Extracts text from legal documents using PyMuPDF or similar tools
2. **Chunking**: Splits laws into semantic units (sections, articles, rules) with metadata
3. **Embedding Generation**: Computes dense vectors using sentence transformers
4. **Export**: Saves `embeddings.npy` (NumPy array) and `docs.json` (metadata)

This process isn't part of the deployed app, which is why the production container stays light.

![Flow](https://github.com/theunknownodysseus/indian-law-rag-pipeline-locallyfetched/blob/main/FLOW%20-NP.png)

---

### Online Service (Deployed)

The FastAPI app does four things:

1. Loads the precomputed embeddings and metadata into memory at startup
2. Encodes incoming questions using a lightweight hashing function (SHA-256 based) into the same vector space
3. Computes cosine similarity via matrix multiplication, filters by topic, selects top matches
4. Calls the Chimera LLM with retrieved context and returns formatted answers

**Why Chimera?** It's available free on OpenRouter, handles long context windows well, and produces coherent legal reasoning without fine-tuning. For a proof-of-concept system, it hits the sweet spot of capability and cost.

---

## Features

### What Makes This Different

- **No Vector Database Runtime**: Pinecone, Weaviate, and similar services are great but add latency and cost. This system runs similarity search in-memory with NumPy, which is fast enough for thousands of documents and costs nothing.

- **Source Attribution**: Every answer includes the top 5 sources used, with similarity scores. Users can verify claims against actual statutes.

- **Topic Routing**: A lightweight LLM call classifies questions before retrieval, improving relevance. "I want to divorce my wife" gets routed to family law, not criminal procedure.

- **Crisis Framing**: Responses assume urgency. They prioritize immediate action items, contact information, and deadlines over theoretical legal discussion.

---

## API Reference

### Base URL

```
https://indian-law-rag-pipeline-production.up.railway.app
```

### Health Check

**Endpoint:** `GET /health`

Returns system status and configuration:

```json
{
  "status": "healthy",
  "model": "tngtech/tng-r1t-chimera:free",
  "documents": 2847,
  "embedding_dim": 512
}
```

### Ask a Legal Question

**Endpoint:** `POST /ask`

**Request:**

```json
{
  "question": "My landlord won't return my security deposit. What can I do?"
}
```

**Response:**

```json
{
  "success": true,
  "answer": "CRISIS SITUATION: Security Deposit Recovery\n\n1. IMMEDIATE ACTION:\n   - Send written notice to landlord via registered post demanding return within 15 days\n   - Keep copies of rent agreement, deposit receipt, possession letter\n   - Document property condition with photos/video\n\n2. LEGAL BASIS:\n   Under Transfer of Property Act and state rent control laws, landlords must return deposits within 30-60 days of lease termination unless legitimate deductions apply...\n\n3. NEXT STEPS:\n   - File complaint with local rent control authority if available\n   - Consider consumer court for compensation (jurisdiction up to ₹1 crore)\n   - Small causes court for civil suit if above ₹20,000...",
  "sources": [
    {
      "source": "transfer_of_property_act_1882",
      "kind": "section",
      "score": 0.89
    },
    {
      "source": "consumer_protection_act_2019",
      "kind": "section",
      "score": 0.84
    }
  ],
  "total_context": 8
}
```

**Fields:**

- `success`: Boolean indicating if the request processed successfully
- `answer`: Crisis-formatted legal guidance ready for display
- `sources`: Top 5 documents used, with source identifier, type (section/article/rule), and similarity score
- `total_context`: Number of chunks retrieved internally (useful for debugging)

---

## Local Development

### Prerequisites

- Python 3.11 or higher
- pip
- Virtual environment tool (venv or similar)

### Setup

Clone the repository:

```bash
git clone https://github.com/theunknownodysseus/indian-law-rag-pipeline.git
cd indian-law-rag-pipeline
```

Create and activate virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set environment variables:

```bash
export OPENROUTER_API_KEY=your_openrouter_api_key_here
```

Ensure data files exist:

The repository should include `embeddings.npy` and `docs.json` in the root directory. These files contain the precomputed embeddings and metadata for the legal corpus.

Run the development server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Access the API documentation at `http://localhost:8000/docs` to test endpoints interactively.

---

## Deployment

### Railway (Current Setup)

The app is configured for Railway's free tier:

1. Connect your GitHub repository to Railway
2. Railway auto-detects the Python/FastAPI service
3. Set `OPENROUTER_API_KEY` in the Railway dashboard under environment variables
4. Ensure `embeddings.npy` and `docs.json` are committed to the repository
5. Railway builds and deploys automatically on push

**Why Railway Works**: Because we ship precomputed embeddings instead of ML libraries, the Docker image stays under 1GB. No GPU needed, no heavy dependencies, no vector database to provision.

### Other Platforms

The app should work on any platform that supports Python 3.11+ and has at least 512MB of RAM:

- **Render**: Similar setup to Railway, free tier available
- **Fly.io**: Works well with the small image size
- **DigitalOcean App Platform**: Minimal configuration required
- **Heroku**: Should work, though less cost-effective than Railway

Just set the `OPENROUTER_API_KEY` environment variable and ensure data files are included in the build.

---

## Technical Decisions

### Why Not Use a Vector Database?

Vector databases like Pinecone, Weaviate, and Qdrant are excellent for production systems, but they add:
- External service dependency and latency
- Additional cost (even free tiers have limits)
- Deployment complexity

For a corpus of ~3,000 legal chunks, NumPy-based cosine similarity runs in under 100ms on a single CPU core. The entire embedding matrix fits comfortably in memory. Unless you're scaling to millions of documents or need real-time updates, the complexity isn't justified.

### Why Chimera?

Several LLMs were tested during development:

- **GPT-4**: Excellent output quality but expensive ($0.03/1K tokens)
- **Claude Sonnet**: Great at legal reasoning but also costly
- **Llama 3**: Good but required self-hosting or paid inference
- **Chimera**: Free tier on OpenRouter, 128K context window, decent instruction-following

For a free-to-use demonstration project, Chimera was the pragmatic choice. The architecture is model-agnostic—swap the OpenRouter endpoint to upgrade.

### Why Crisis Format?

Traditional legal chatbots respond with "According to Section 498A of IPC..." which is accurate but unhelpful in an emergency. The crisis format prioritizes:

1. What to do right now
2. What evidence to preserve
3. Who to contact
4. What deadlines matter
5. What legal provisions support these actions

This doesn't replace legal counsel, but it gives people a starting point when they need one immediately.

### Why Offline Embedding Generation?

Initial deployment attempts with sentence-transformers and PyTorch failed because:
- The Docker image exceeded 4GB (Railway's limit)
- Build times exceeded 15 minutes (deployment timeout)
- Runtime memory usage was high even with small models

Moving embedding generation offline reduced the production image to ~800MB and startup time to under 5 seconds. The tradeoff is that updating the corpus requires regenerating embeddings, but for a relatively static legal corpus, this happens infrequently enough that it's acceptable.

---

## Limitations and Known Issues

### Current Constraints

- **Response Time**: End-to-end latency averages 15-20 seconds due to OpenRouter API calls. This is acceptable for research queries but slow for real-time chat.

- **No Case Law**: The system only includes statutory text, not Supreme Court or High Court judgments. Case law provides crucial interpretation and precedent that's currently missing.

- **Static Corpus**: Updating laws requires regenerating embeddings offline. There's no mechanism to dynamically add new legislation.

- **English Only**: All content is in English. Many Indian statutes have Hindi or regional language versions that aren't covered.

- **No Authentication**: The API is publicly accessible without rate limiting. This works for a demo but isn't production-ready.

### Accuracy Considerations

This system is a legal information tool, not legal advice. It can:
- Surface relevant statutory provisions
- Provide general guidance on legal procedures
- Point users toward appropriate resources

It cannot:
- Replace consultation with a qualified lawyer
- Provide case-specific legal advice
- Guarantee accuracy for complex or ambiguous situations

Always verify important legal information with professional counsel.

---

## Future Improvements

### Planned Features

**Case Law Integration**: Add Supreme Court and High Court judgments with separate metadata tags. This would require parsing PDFs from judis.nic.in and India Kanoon, then distinguishing case citations from statutory references in retrieval.

**Authentication & Rate Limiting**: Implement API keys with usage quotas to prevent abuse while keeping research access free.

**Faster Models**: Experiment with Llama 3.1 8B or Gemini Flash for sub-5-second response times. The current 15-20 second latency is the main UX bottleneck.

**Multi-language Support**: Add Hindi and major regional language versions of key statutes. This requires translating embeddings or maintaining separate indices per language.

**Conversation Memory**: Track multi-turn conversations to handle follow-up questions like "What's the punishment for that?" without needing to re-specify context.

**Citation Verification**: Add links to original source PDFs or government websites so users can read full sections in context.

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| **API Framework** | FastAPI | Async support, automatic OpenAPI docs, type safety |
| **Language** | Python 3.11+ | Ecosystem for ML/NLP, NumPy performance |
| **LLM** | Chimera (OpenRouter) | Free tier, good context window, legal reasoning |
| **Embeddings** | Precomputed NumPy | No runtime dependencies, fast similarity search |
| **Deployment** | Railway | Free tier, auto-deploy from GitHub, simple config |
| **Retrieval** | Cosine Similarity | Sufficient for <10K docs, no database overhead |

---

## Contributing

This is a research project and proof-of-concept. If you want to improve it:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-idea`)
3. Make your changes with clear commit messages
4. Test locally with `pytest` (tests are in `/tests` if they exist)
5. Submit a pull request with a description of what you've added

Particularly valuable contributions:
- Adding more legal corpora (state laws, case law)
- Improving retrieval accuracy
- Reducing response latency
- Adding multi-language support

---

## License

This project's code is available under the MIT License. However:

- **Legal corpus**: Indian statutes are public domain as government works
- **OpenRouter/Chimera**: Subject to OpenRouter's terms of service
- **Dependencies**: Each library has its own license (see `requirements.txt`)

Review all upstream licenses before commercial use.

---

## Acknowledgments

- Legal corpus sourced from [IndianKanoon](https://indiankanoon.org/) and [Ministry of Law and Justice](https://legislative.gov.in/)
- Built with [FastAPI](https://fastapi.tiangolo.com/) by Sebastián Ramírez
- LLM access via [OpenRouter](https://openrouter.ai/)
- Inspired by legal aid initiatives and the need for accessible legal information

---

## Contact

For questions, suggestions, or issues:

- **GitHub Issues**: [Open an issue](https://github.com/theunknownodysseus/indian-law-rag-pipeline-locallyfetched/issues)
- **Repository**: [github.com/theunknownodysseus/indian-law-rag-pipeline](https://github.com/theunknownodysseus/indian-law-rag-pipeline-locallyfetched)

Built by [Varun](https://github.com/theunknownodysseus)
---

**Disclaimer**: This system provides general legal information, not legal advice. It cannot replace consultation with a qualified lawyer. For specific legal issues, always consult appropriate legal counsel.
