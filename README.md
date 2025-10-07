---

# AI-Powered Knowledge Base Search & Enrichment

This project is a complete, API-driven application that allows users to build an intelligent knowledge base. It uses a Retrieval-Augmented Generation (RAG) pipeline to provide answers to natural language questions based on user-uploaded documents. The system can dynamically enrich its knowledge from external sources and includes a feedback loop for continuous improvement.

## Features

-   **📄 Document Upload:** Users can upload documents (`.pdf`, `.txt`, `.md`) via a simple web interface.
-   **🧠 Vector Indexing:** Uploaded documents are automatically chunked, vectorized (using OpenAI embeddings), and stored in a persistent ChromaDB vector store.
-   **💬 Natural Language Search:** A sophisticated RAG pipeline provides answers to user questions grounded in the provided documents.
-   **✨ Dynamic Enrichment:** If the local knowledge base is insufficient to answer a query, the system temporarily fetches relevant information from Wikipedia to provide a comprehensive answer **without permanently altering the curated knowledge base**.
-   **📊 Structured Responses:** The AI generates detailed, structured JSON responses, including the answer, a confidence score, and suggestions for knowledge base improvement.
-   **⭐ User Feedback:** A complete feedback loop allows users to rate answer quality on a 1-10 scale. This feedback is logged to a CSV file for future analysis and pipeline tuning.
-   **🌐 Simple Web Interface:** A self-contained HTML frontend for easy interaction with the application.


## Setup and Running the Application

Follow these steps to get the application running on your local machine.

### 1. Prerequisites

-   Python 3.8+
-   `pip` for package management

### 2. Clone the Repository

Clone this project to your local machine.

```bash
git clone <your-repository-url>
cd <your-project-directory>
```

### 3. Create the Environment File

Create a file named `.env` in the root of the project directory. This file will hold your secret keys and configuration. Copy the following into it and add your values.

**`.env` file:**
```env
# Required: Your secret API key from OpenAI.
OPENAI_API_KEY="sk-..."
MAX_FILE_SIZE="25" # In megabytes
ALLOWED_ORIGINS="http://localhost:8000,http://127.0.0.1:8000"
PORT="8000"
```

### 4. Install Dependencies

Install all the required Python packages using `pip`.

```bash
pip install -r requirements.txt
```

### 5. Run the Server

Use `uvicorn` to run the FastAPI application. The `--reload` flag will automatically restart the server when you make code changes.

```bash
uvicorn main:app --reload
```

### 6. Access the Application

Once the server is running, open your web browser and navigate to:

**`http://127.0.0.1:8000`**

You should now see the web interface and be able to upload documents and ask questions.

---

## API Endpoints

The application exposes the following RESTful API endpoints:

### 1. Upload Document

- **Method:** `POST`
- **Path:** `/upload/document`
- **Request Body:** `multipart/form-data` with a `file` field.
- **Description:** Uploads a single document. The file is saved, indexed, and added to the vector store.
- **Success Response (200 OK):**
  ```json
  {
    "message": "File uploaded and indexed successfully",
    "filename": "example.pdf"
  }
  ```

### 2. Search Document

- **Method:** `POST`
- **Path:** `/search/doc`
- **Request Body:**
  ```json
  {
    "query": "What is the main topic of the document?"
  }
  ```
- **Description:** Performs a RAG search. It combines context from the local vector store and enrichment from Wikipedia to generate an answer.
- **Success Response (200 OK):**
  ```json
  {
    "answer": "The main topic is the implementation of Retrieval-Augmented Generation.",
    "confidence": 0.9,
    "missing_info": "",
    "enrichment_suggestion": "",
    "search_id": "c4e1b8a1-3b1a-4b1e-8e1a-9a1b1e1a1b1e"
  }
  ```

### 3. Submit Feedback

- **Method:** `POST`
- **Path:** `/feedback`
- **Request Body:**
  ```json
  {
    "search_id": "c4e1b8a1-3b1a-4b1e-8e1a-9a1b1e1a1b1e",
    "query": "What is the main topic of the document?",
    "answer": "The main topic is the implementation of Retrieval-Augmented Generation.",
    "rating": 9,
    "comment": "The answer was very relevant and helpful."
  }
  ```
- **Description:** Submits user feedback for a specific search result. The data is logged to `feedback.csv`.
- **Success Response (201 Created):**
  ```json
  {
    "message": "Feedback received successfully. Thank you!"
  }
  ```

---

## Tradeoffs and Limitations

This application was built as a robust prototype. Several tradeoffs were made for the sake of simplicity and rapid development:

- **Local Storage:** All data (uploaded files, vector index, feedback logs) is stored on the local filesystem. This is not suitable for a scalable, multi-user production environment where a cloud bucket (like S3), a managed vector database (like Pinecone), and a proper database would be required.
- **Synchronous Processing:** Document uploads and indexing are handled synchronously. For large files, this could block the server and lead to request timeouts. A production system would use a background worker queue (like Celery) for processing.
- **No User Authentication:** The API is open and has no concept of users or permissions. Anyone with access can upload, search, and submit feedback.
- **Basic Frontend:** The UI is a minimal, single HTML file designed for basic functionality and demonstration.
- **Configuration in Code:** While the OpenAI key is in `.env`, other parameters like the LLM model name, chunk size, and retriever settings are hardcoded in the service classes. These should ideally be loaded from a configuration file or environment variables. The `.env` variables `MAX_FILE_SIZE`, `ALLOWED_ORIGINS`, and `PORT` are included as best practice placeholders but are not yet fully integrated into the FastAPI application's configuration.

## Future Improvements

- **Production-Grade Infrastructure:** Migrate storage to cloud services (S3, managed PostgreSQL, Pinecone/Weaviate).
- **Asynchronous Indexing:** Implement a task queue (e.g., Celery with Redis) to process document uploads in the background.
- **User Authentication:** Add an authentication layer (e.g., OAuth2) to manage users and secure the knowledge base.
- **Advanced Feedback Analysis:** Build a dashboard to analyze the `feedback.csv` data, allowing for easy identification of weak points in the RAG pipeline.
- **Automated Retraining/Fine-Tuning:** Use the collected feedback data to create evaluation datasets and potentially fine-tune a model for better performance.
- **Enhanced Frontend:** Develop a more robust frontend using a framework like React or Vue.js for a better user experience.
