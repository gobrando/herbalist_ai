import chromadb
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from sentence_transformers import SentenceTransformer
import os

class VectorStore:
    """
    Handles vector storage and similarity search using ChromaDB.
    """
    
    def __init__(self, 
                 collection_name: str = "herbalism_knowledge",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 persist_directory: str = "./chroma_db"):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: Name of the sentence transformer model to use
            persist_directory: Directory to persist the ChromaDB data
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.logger = logging.getLogger(__name__)
        
        # Initialize sentence transformer
        self.logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Try to get existing collection or create new one
        try:
            self.collection = self.client.get_collection(name=collection_name)
            self.logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Herbalism knowledge base"}
            )
            self.logger.info(f"Created new collection: {collection_name}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Create embedding for given text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document dictionaries with 'content', 'metadata' keys
        """
        self.logger.info(f"Adding {len(documents)} documents to vector store")
        
        # Prepare data for ChromaDB
        embeddings = []
        ids = []
        metadatas = []
        documents_text = []
        
        for i, doc in enumerate(documents):
            # Create embedding
            embedding = self.embed_text(doc['content'])
            embeddings.append(embedding)
            
            # Create unique ID
            doc_id = f"doc_{i}_{hash(doc['content'][:100])}"
            ids.append(doc_id)
            
            # Prepare metadata
            metadata = doc.get('metadata', {})
            metadatas.append(metadata)
            
            # Store document text
            documents_text.append(doc['content'])
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=documents_text,
            metadatas=metadatas,
            ids=ids
        )
        
        self.logger.info(f"Successfully added {len(documents)} documents")
    
    def similarity_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of similar documents with metadata
        """
        # Create query embedding
        query_embedding = self.embed_text(query)
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        documents = []
        if results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                doc = {
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'][0] else {},
                    'distance': results['distances'][0][i] if results['distances'] else None
                }
                documents.append(doc)
        
        self.logger.info(f"Found {len(documents)} similar documents for query: {query[:50]}...")
        return documents
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Dictionary with collection metadata
        """
        count = self.collection.count()
        return {
            'name': self.collection_name,
            'document_count': count,
            'persist_directory': self.persist_directory
        }