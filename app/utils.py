from dotenv import load_dotenv
import os

# Load environment variables from .env file if it exists
load_dotenv()

from pydantic import BaseModel
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import Document, TextNode
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.core.query_engine import CitationQueryEngine
from dataclasses import dataclass
import pdfplumber
import re
from pathlib import Path
from typing import Optional
from types import MethodType

key = os.getenv('OPENAI_API_KEY')
if not key:
    raise ValueError(
        "OPENAI_API_KEY environment variable is not set. "
        "Please set it in your environment or create a .env file with OPENAI_API_KEY=your_key"
    )

@dataclass
class Input:
    query: str
    file_path: str

class Citation(BaseModel):
    source: str
    text: str
    page: Optional[int] = None
    subsection_id: Optional[str] = None
    section_name: Optional[str] = None

class Output(BaseModel):
    query: str
    response: str
    citations: list[Citation]

class DocumentService:
    def __init__(self, pdf_path: str = "docs/laws.pdf"):
        self.pdf_path = pdf_path
    
    def create_documents(self) -> list[Document]:
        """
        Parse the PDF and extract laws into Document objects with metadata.
        Each law section/subsection becomes a Document with metadata including:
        - section_id: e.g., "5"
        - subsection_id: e.g., "5.1.1"
        - section_name: e.g., "Trials by combat"
        - page: page number in PDF
        """
        docs = []
        
        with pdfplumber.open(self.pdf_path) as pdf:
            current_section_id = None
            current_subsection_id = None
            current_section_name = None
            current_text = []
            current_page = None
            
            # Pattern to match section headers like "5", "5.1", "5.1.1"
            section_pattern = re.compile(r'^(\d+(?:\.\d+)*)\s*(.*?)$')
            # Pattern to match subsection headers
            subsection_pattern = re.compile(r'^(\d+(?:\.\d+)+)\s+(.+)$')
            
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if not text:
                    continue
                
                lines = text.split('\n')
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check if this is a section header (starts with number)
                    section_match = section_pattern.match(line)
                    if section_match:
                        # Save previous section if exists
                        if current_text and current_section_id:
                            doc_text = '\n'.join(current_text).strip()
                            if doc_text:
                                docs.append(Document(
                                    text=doc_text,
                                    metadata={
                                        "section_id": current_section_id,
                                        "subsection_id": current_subsection_id or current_section_id,
                                        "section_name": current_section_name or f"Section {current_section_id}",
                                        "page": current_page or page_num
                                    }
                                ))
                        
                        # Start new section
                        current_section_id = section_match.group(1)
                        section_name_part = section_match.group(2).strip()
                        
                        # Determine if this is a subsection (has dots) or main section
                        if '.' in current_section_id:
                            # This is a subsection
                            parts = current_section_id.split('.')
                            if len(parts) == 2:
                                # Parent section like 5.1
                                current_subsection_id = current_section_id
                                current_section_name = section_name_part if section_name_part else f"Section {current_section_id}"
                            else:
                                # Deeper subsection like 5.1.1
                                current_subsection_id = current_section_id
                                current_section_name = section_name_part if section_name_part else None
                        else:
                            # Main section
                            current_subsection_id = None
                            current_section_name = section_name_part if section_name_part else f"Section {current_section_id}"
                        
                        current_text = []
                        current_page = page_num
                    else:
                        # Continuation of current section
                        if current_section_id:
                            current_text.append(line)
            
            # Save last section
            if current_text and current_section_id:
                doc_text = '\n'.join(current_text).strip()
                if doc_text:
                    docs.append(Document(
                        text=doc_text,
                        metadata={
                            "section_id": current_section_id,
                            "subsection_id": current_subsection_id or current_section_id,
                            "section_name": current_section_name or f"Section {current_section_id}",
                            "page": current_page or len(pdf.pages)
                        }
                    ))
        
        # Post-process: ensure section_name is populated from parent if missing
        section_name_map = {}
        for doc in docs:
            section_id = doc.metadata.get("section_id")
            subsection_id = doc.metadata.get("subsection_id")
            section_name = doc.metadata.get("section_name")
            
            if section_name and not section_name.startswith("Section"):
                section_name_map[section_id] = section_name
                if subsection_id:
                    section_name_map[subsection_id] = section_name
        
        # Fill in missing section names from parent sections
        for doc in docs:
            if not doc.metadata.get("section_name") or doc.metadata.get("section_name", "").startswith("Section"):
                section_id = doc.metadata.get("section_id")
                subsection_id = doc.metadata.get("subsection_id")
                
                # Try to get from subsection first, then section
                if subsection_id and subsection_id in section_name_map:
                    doc.metadata["section_name"] = section_name_map[subsection_id]
                elif section_id and section_id in section_name_map:
                    doc.metadata["section_name"] = section_name_map[section_id]
        
        return docs

class QdrantService:
    def __init__(self, k: int = 2):
        self.index = None
        self.k = k
    
    def connect(self) -> None:
        client = qdrant_client.QdrantClient(location=":memory:")

        if not hasattr(client, "search") and hasattr(client, "query_points"):
            def _search(self, collection_name: str, query_vector, limit: int, query_filter=None, **kwargs):
                response = self.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    limit=limit,
                    query_filter=query_filter,
                    with_payload=True,
                    **kwargs,
                )
                return response.points
            client.search = MethodType(_search, client)

        vstore = QdrantVectorStore(client=client, collection_name='temp')

        # Configure embedding model and LLM
        embed_model = OpenAIEmbedding(api_key=key)
        llm = OpenAI(api_key=key, model="gpt-4o-mini")
        
        # Set global settings
        Settings.embed_model = embed_model
        Settings.llm = llm

        storage_context = StorageContext.from_defaults(vector_store=vstore)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vstore,
            storage_context=storage_context,
            embed_model=embed_model
        )

    def load(self, docs: Optional[list[Document]] = None):
        """
        Load documents into the index. If docs is None, create documents from PDF.
        """
        if docs is None:
            doc_service = DocumentService()
            docs = doc_service.create_documents()
        
        # Convert Documents to Nodes for insertion
        nodes = []
        for doc in docs:
            node = TextNode(
                text=doc.text,
                metadata=doc.metadata
            )
            nodes.append(node)
        
        self.index.insert_nodes(nodes)
    
    def query(self, query_str: str) -> Output:
        """
        Query the index using CitationQueryEngine and return formatted Output.
        """
        if self.index is None:
            raise ValueError("Index not initialized. Call connect() first.")
        
        # Initialize CitationQueryEngine with similarity_top_k = self.k
        query_engine = CitationQueryEngine.from_args(
            self.index,
            similarity_top_k=self.k,
            citation_chunk_size=512
        )
        
        # Run the query
        response = query_engine.query(query_str)
        
        # Extract citations from source nodes
        citations = []
        if hasattr(response, 'source_nodes') and response.source_nodes:
            for node in response.source_nodes:
                metadata = node.metadata if hasattr(node, 'metadata') else {}
                node_text = node.text if hasattr(node, 'text') else str(node)
                
                # Extract metadata fields
                source_id = metadata.get('subsection_id') or metadata.get('section_id', 'Unknown')
                page = metadata.get('page')
                subsection_id = metadata.get('subsection_id')
                section_name = metadata.get('section_name')
                
                citations.append(Citation(
                    source=source_id,
                    text=node_text,
                    page=page,
                    subsection_id=subsection_id,
                    section_name=section_name
                ))
        
        # Get response text
        response_text = str(response) if response else ""
        
        return Output(
            query=query_str,
            response=response_text,
            citations=citations
        )
       

if __name__ == "__main__":
    # Example workflow
    doc_serivce = DocumentService() # implemented
    docs = doc_serivce.create_documents() # NOT implemented

    index = QdrantService() # implemented
    index.connect() # implemented
    index.load() # implemented

    index.query("what happens if I steal?") # NOT implemented





