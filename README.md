# RAG Pipeline API - Game of Thrones Laws

This repository contains a RAG (Retrieval-Augmented Generation) pipeline API that allows querying a database of laws from the fictional series "Game of Thrones". The API uses LlamaIndex, Qdrant vector store, and OpenAI embeddings/LLM to provide semantic search over the laws PDF document.

## Server Repository

### Overview

The server implements a RAG pipeline that:
- Parses `docs/laws.pdf` into structured sections and subsections
- Creates vector embeddings for each law section
- Provides semantic search capabilities via a FastAPI endpoint
- Returns answers with citations including page numbers, section IDs, and section names

### Prerequisites

- Python 3.11+
- Docker (for containerized deployment)
- OpenAI API key
- API key for authentication (you can set your own)

### Local Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables:**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export API_KEY="your-api-key-for-auth"
   ```

3. **Run the server:**
   ```bash
   uvicorn app.main:app --host localhost --port 8000
   ```

   The server will start on `http://localhost:8000`

### Docker Deployment

1. **Build the Docker image:**
   ```bash
   docker build -t norm-rag-api .
   ```

2. **Run the container:**
   ```bash
   docker run -d \
     -p 8000:80 \
     -e OPENAI_API_KEY="your-openai-api-key" \
     -e API_KEY="your-api-key-for-auth" \
     --name norm-rag-api \
     norm-rag-api
   ```

   The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, you can access:
- **Interactive API docs:** `http://localhost:8000/docs` (Swagger UI)
- **Alternative docs:** `http://localhost:8000/redoc` (ReDoc)
- **Health check:** `http://localhost:8000/health`

### API Endpoints

#### POST `/query`

Query the laws database using RAG.

**Headers:**
- `X-API-Key`: Your API key (required)

**Request Body:**
```json
{
  "query": "What happens if I steal?",
  "k": 3
}
```

**Parameters:**
- `query` (string, required): The query string to search for
- `k` (integer, optional): Number of similar vectors to return (default: 2, max: 20)

**Response:**
```json
{
  "query": "What happens if I steal?",
  "response": "According to the laws, theft is punishable by hanging...",
  "citations": [
    {
      "source": "5.1.1",
      "text": "Theft is punishable by hanging...",
      "page": 12,
      "subsection_id": "5.1.1",
      "section_name": "Trials by combat"
    }
  ]
}
```

**Response Fields:**
- `query`: The original query string
- `response`: The LLM-generated answer based on retrieved laws
- `citations`: Array of citation objects with:
  - `source`: Section or subsection ID (e.g., "5.1.1")
  - `text`: The relevant text snippet from the law
  - `page`: Page number in the PDF (optional)
  - `subsection_id`: Subsection identifier (e.g., "5.1.1") (optional)
  - `section_name`: Name of the section (e.g., "Trials by combat") (optional)

### Example Usage

#### Using curl:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What happens if I steal from the Sept?",
    "k": 3
  }'
```

#### Using Python:

```python
import requests

url = "http://localhost:8000/query"
headers = {
    "X-API-Key": "your-api-key",
    "Content-Type": "application/json"
}
data = {
    "query": "What happens if I steal from the Sept?",
    "k": 3
}

response = requests.post(url, json=data, headers=headers)
print(response.json())
```

### Design Choices and Assumptions

1. **PDF Parsing:** The PDF is parsed by detecting section headers that start with numbers (e.g., "5", "5.1", "5.1.1"). Each section becomes a separate document in the vector store.

2. **Section Names:** Section names are extracted from headers when available. If a subsection doesn't have a name, it inherits from its parent section.

3. **Vector Store:** Uses Qdrant in-memory vector store for simplicity. In production, you might want to use a persistent Qdrant instance.

4. **Initialization:** The index is built once at startup to avoid re-embedding the PDF on every request, improving performance.

5. **Authentication:** Simple API key authentication via header. In production, consider more robust authentication mechanisms.

6. **Error Handling:** The API includes basic error handling for invalid queries, missing API keys, and service initialization failures.

### Limitations

- The PDF parsing relies on consistent formatting. Irregular PDFs may require adjustments to the parsing logic.
- The in-memory Qdrant store means the index is rebuilt on each container restart.
- Query length is limited to 1000 characters.
- The `k` parameter is limited to a maximum of 20 to prevent performance issues.

## Client Repository 

In the `frontend` folder you'll find a light NextJS app with its own README including instructions to run. Your task here is to build a minimal client experience that utilizes the service built in part 1.
