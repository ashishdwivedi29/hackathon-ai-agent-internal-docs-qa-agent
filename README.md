# Internal Docs Q&A Agent

## Project Description
A real-time AI-powered Q&A agent for internal documents, built for the hackathon. Users can upload documents (PDFs, DOCX, Markdown), ask natural language questions, and get instant answers sourced from the documents.

## Tech Stack
-   **Backend**: Python, LangChain, Google Gemini API, FAISS
-   **Frontend**: Streamlit
-   **Data**: PDFs, DOCX, .txt, .md

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd InternalDocsQAAgent
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up Google Gemini API Key:**
    * Go to [Google AI Studio](https://aistudio.google.com/app/apikey) to generate a free API key for Gemini.
    * Create a `.env` file in the root of this project (`InternalDocsQAAgent/`) and add your API key:
        ```
        GOOGLE_API_KEY="YOUR_API_KEY_HERE"
        ```
        *(Make sure `.env` is listed in your `.gitignore`!)*

## How to Run

1.  Place your internal documents (PDFs, DOCX, TXT, MD) into the `docs/` directory.
2.  Run the document ingestion script:
    ```bash
    python backend/ingest.py
    ```
3.  Start the Streamlit application:
    ```bash
    streamlit run app.py
    ```
    This will open the application in your web browser.

## Demo Queries
(Add 4-5 example questions here that work well with your sample documents)

## Architecture Diagram
(Will be added here later)

## Team Members
-   [Your Name/Team Name]