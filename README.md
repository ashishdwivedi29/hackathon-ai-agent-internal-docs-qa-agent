# 🧠 Internal Docs Q&A Agent

A real-time **AI-powered Q&A assistant** for internal documents — built for a hackathon to help users interact with their documents using **natural language**. Upload your internal files (PDFs, DOCX, Markdown, TXT), ask questions, and get instant, context-rich answers directly sourced from your documents.

---

## 🚀 Project Highlights

### ✅ Features Implemented (Day 1 & 2)

- 📄 **Document Ingestion**  
  Upload and process PDFs, DOCX, TXT, and Markdown files.

- 🧠 **Intelligent Indexing**  
  Uses the **Google Gemini API** to convert documents into high-quality embeddings.

- ⚡ **Efficient Search**  
  Embeddings are stored in a **FAISS** vector database for lightning-fast retrieval.

- 💬 **Natural Language Q&A**  
  Ask questions like you're chatting with an assistant — no technical jargon needed.

- 🧩 **Contextual Answering**  
  Answers are generated from **your** documents, using Gemini's deep understanding.

---

## 🧰 Tech Stack

| Layer        | Tools Used                            |
|--------------|----------------------------------------|
| **Backend**  | Python, LangChain, FAISS, Google Gemini API |
| **Frontend** | Streamlit                             |
| **Data**     | PDFs, DOCX, TXT, Markdown             |

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository:

```bash
git clone https://github.com/ashishdwivedi29/hackathon-ai-agent-internal-docs-qa-agent.git
cd internal-docs-qa-agent
````

### 2️⃣ Create a virtual environment:
s
```bash
python -m venv venv
# For Windows
venv\Scripts\activate
# For macOS/Linux
source venv/bin/activate
```

### 3️⃣ Install dependencies:

```bash
pip install -r requirements.txt
```

### 4️⃣ Set up Google Gemini API key:

* Visit **[Google AI Studio](https://makersuite.google.com/app/apikey)** and generate your free API key.
* Create a `.env` file in the project root and add the following:

```env
GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```

✅ *Note: `.env` is already ignored via `.gitignore`*

---

## ▶️ How to Run

1. Place your internal documents in the `docs/` directory.

2. Run the document ingestion pipeline:

```bash
python backend/ingest.py
```

3. Launch the Streamlit application:

```bash
streamlit run app.py
```

Your AI Q\&A agent will now open in your default browser.

---

## 💡 Demo Queries

Ask your assistant real questions like:

* “What is the company’s refund policy?”
* “How do I request time off?”
* “Can you summarize the Q3 financial report?”
* “What are the onboarding steps for new employees?”
* “What benefits are offered to full-time staff?”

*🔍 Results depend on the content you upload in `docs/`.*

---

## 🧱 Architecture Diagram

📌 *Coming soon*

---

## 👥 Team Members

* **Ashish Dwivedi**
* **Vedant Bhosale**

---

> Built with ❤️ and Gemini API in a 2-day hackathon.

```


