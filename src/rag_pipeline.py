import logging
import os
from typing import List, Dict, Any, Optional
import time

from .document_processor import DocumentProcessor
from .vector_store import VectorStore
from .llm_interface import LLMInterface

class RAGPipeline:
    """
    Main RAG pipeline that orchestrates document processing, vector storage, and response generation.
    """
    
    def __init__(self,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 llm_type: str = "local",
                 llm_model: str = "gpt-3.5-turbo",
                 temperature: float = 0.7,
                 max_tokens: int = 1000,
                 top_k_results: int = 5):
        """
        Initialize the RAG pipeline.
        
        Args:
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
            embedding_model: Model for creating embeddings
            llm_type: Type of LLM to use ('openai', 'ollama', 'local')
            llm_model: Specific model name
            temperature: LLM temperature setting
            max_tokens: Maximum tokens for LLM response
            top_k_results: Number of top similar chunks to retrieve
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k_results = top_k_results
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.logger.info("Initializing RAG pipeline components...")
        
        # Document processor
        self.doc_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Vector store
        self.vector_store = VectorStore(
            embedding_model=embedding_model,
            persist_directory="./chroma_db"
        )
        
        # LLM interface
        self.llm = LLMInterface(
            model_type=llm_type,
            model_name=llm_model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        self.logger.info("RAG pipeline initialized successfully")
    
    def process_documents(self, file_paths: List[str]) -> None:
        """
        Process documents and add them to the vector store.
        
        Args:
            file_paths: List of paths to documents to process
        """
        self.logger.info(f"Processing {len(file_paths)} documents...")
        
        all_documents = []
        
        for file_path in file_paths:
            self.logger.info(f"Processing file: {file_path}")
            
            # Load and clean document
            text = self.doc_processor.load_document(file_path)
            cleaned_text = self.doc_processor.clean_text(text)
            
            # Split into chunks
            chunks_data = self.doc_processor.chunk_text(cleaned_text)
            chunks = [chunk['content'] for chunk in chunks_data]
            
            # Convert chunks to document format
            for i, chunk in enumerate(chunks):
                doc = {
                    'content': chunk,
                    'metadata': {
                        'source_file': os.path.basename(file_path),
                        'chunk_index': i,
                        'file_path': file_path
                    }
                }
                all_documents.append(doc)
        
        # Add all documents to vector store
        if all_documents:
            self.vector_store.add_documents(all_documents)
            self.logger.info(f"Successfully processed and stored {len(all_documents)} document chunks")
        else:
            self.logger.warning("No documents were processed")
    
    def query(self, question: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Process a user query and generate a response.
        
        Args:
            question: User's question
            conversation_history: Previous conversation messages
            
        Returns:
            Dictionary containing response and metadata
        """
        start_time = time.time()
        self.logger.info(f"Processing query: {question[:100]}...")
        
        try:
            # Retrieve relevant documents
            relevant_docs = self.vector_store.similarity_search(
                query=question,
                top_k=self.top_k_results
            )
            
            if not relevant_docs:
                return {
                    'response': "I couldn't find any relevant information in my knowledge base to answer your question about herbalism. Could you try rephrasing your question?",
                    'sources': [],
                    'processing_time': time.time() - start_time
                }
            
            # Prepare context from retrieved documents
            context_parts = []
            sources = []
            
            for i, doc in enumerate(relevant_docs):
                context_parts.append(f"Source {i+1}: {doc['content']}")
                sources.append({
                    'source_file': doc['metadata'].get('source_file', 'Unknown'),
                    'chunk_index': doc['metadata'].get('chunk_index', 0),
                    'relevance_score': 1.0 - (doc.get('distance', 0) or 0)
                })
            
            context = "\n\n".join(context_parts)
            
            # Build prompt
            prompt = self._build_prompt(question, context, conversation_history)
            
            # Generate response
            response = self.llm.generate_response(prompt)
            
            processing_time = time.time() - start_time
            self.logger.info(f"Query processed in {processing_time:.2f} seconds")
            
            return {
                'response': response,
                'sources': sources,
                'processing_time': processing_time,
                'context_used': len(relevant_docs)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return {
                'response': f"I apologize, but I encountered an error while processing your question: {str(e)}",
                'sources': [],
                'processing_time': time.time() - start_time,
                'error': str(e)
            }
    
    def _build_prompt(self, question: str, context: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Build the prompt for the LLM.
        
        Args:
            question: User's question
            context: Retrieved context
            conversation_history: Previous conversation
            
        Returns:
            Formatted prompt string
        """
        # Build conversation context
        conversation_context = ""
        if conversation_history:
            conversation_context = "\n\nPrevious conversation:\n"
            for msg in conversation_history[-5:]:  # Last 5 messages
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                conversation_context += f"{role.capitalize()}: {content}\n"
        
        prompt = f"""You are an expert herbalist with deep knowledge of traditional and modern herbal medicine. You provide helpful, accurate, and safe advice about herbs and natural remedies.

IMPORTANT GUIDELINES:
- Base your answers primarily on the provided context from the herbalism knowledge base
- Always emphasize safety and recommend consulting healthcare professionals for serious conditions
- Be clear about the traditional vs. modern uses of herbs
- If you're unsure about something, say so rather than guessing
- Provide practical, actionable advice when appropriate

Context from herbalism knowledge base:
{context}
{conversation_context}

Current question: {question}

Please provide a comprehensive, helpful response based on the herbalism knowledge provided. If the context doesn't contain enough information to fully answer the question, acknowledge this and provide what information you can from your general herbalism knowledge, but clearly distinguish between information from the knowledge base and your general knowledge."""

        return prompt
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG pipeline.
        
        Returns:
            Dictionary with pipeline statistics
        """
        vector_info = self.vector_store.get_collection_info()
        
        return {
            'pipeline_status': 'active',
            'vector_store': vector_info,
            'llm_model': self.llm.model_name,
            'llm_type': self.llm.model_type,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'top_k_results': self.top_k_results
        }
    
    def clear_vector_store(self) -> None:
        """
        Clear all documents from the vector store.
        """
        # Since ChromaDB doesn't have a clear method, we'll reinitialize
        self.vector_store = VectorStore(
            embedding_model="all-MiniLM-L6-v2",
            persist_directory="./chroma_db"
        )
        self.logger.info("Vector store cleared")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on all components.
        
        Returns:
            Dictionary with health status
        """
        health = {
            'status': 'healthy',
            'components': {
                'document_processor': 'healthy',
                'vector_store': 'healthy',
                'llm': 'healthy'
            },
            'timestamp': time.time()
        }
        
        try:
            # Test vector store
            vector_info = self.vector_store.get_collection_info()
            if vector_info['document_count'] == 0:
                health['components']['vector_store'] = 'warning - no documents'
                health['status'] = 'warning'
        except Exception as e:
            health['components']['vector_store'] = f'error - {str(e)}'
            health['status'] = 'error'
        
        try:
            # Test LLM
            test_response = self.llm.generate_response("Hello")
            if not test_response or len(test_response.strip()) == 0:
                health['components']['llm'] = 'warning - empty response'
                if health['status'] != 'error':
                    health['status'] = 'warning'
        except Exception as e:
            health['components']['llm'] = f'error - {str(e)}'
            health['status'] = 'error'
        
        return health