import re
import os
from typing import List, Dict, Any
import logging

class DocumentProcessor:
    """
    Handles document processing including loading, cleaning, and chunking text.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(__name__)
    
    def load_document(self, file_path: str) -> str:
        """
        Load a text document from file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Raw text content of the document
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
            self.logger.info(f"Loaded document: {file_path} ({len(content)} characters)")
            return content
        except Exception as e:
            self.logger.error(f"Error loading document {file_path}: {str(e)}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Clean the text by removing unnecessary formatting and standardizing whitespace.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Remove Project Gutenberg header and footer
        start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
        end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
        
        start_idx = text.find(start_marker)
        if start_idx != -1:
            # Find the end of the start marker line
            start_idx = text.find('\n', start_idx) + 1
            text = text[start_idx:]
        
        end_idx = text.find(end_marker)
        if end_idx != -1:
            text = text[:end_idx]
        
        # Remove page numbers in square brackets
        text = re.sub(r'\[Page \d+\]', '', text)
        text = re.sub(r'\[\d+\]', '', text)
        
        # Remove excessive whitespace and normalize line breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove transcriber notes
        text = re.sub(r'Transcriber.*?notes?:.*?\n\n', '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove table of contents markers
        text = re.sub(r'_+', '', text)
        
        return text.strip()
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks for better retrieval.
        
        Args:
            text: Cleaned text to chunk
            
        Returns:
            List of dictionaries containing chunk information
        """
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        chunk_id = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If adding this paragraph would exceed chunk size, save current chunk
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                chunks.append({
                    'id': chunk_id,
                    'content': current_chunk.strip(),
                    'length': len(current_chunk),
                    'start_char': len(''.join([c['content'] for c in chunks])),
                })
                chunk_id += 1
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append({
                'id': chunk_id,
                'content': current_chunk.strip(),
                'length': len(current_chunk),
                'start_char': len(''.join([c['content'] for c in chunks])),
            })
        
        self.logger.info(f"Created {len(chunks)} chunks from document")
        return chunks
    
    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """
        Extract metadata from the document.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary containing metadata
        """
        metadata = {
            'total_length': len(text),
            'word_count': len(text.split()),
            'title': 'Herbal Simples Approved for Modern Uses of Cure',
            'author': 'W. T. Fernie, M.D.',
            'source': 'Project Gutenberg',
            'subject': 'Herbalism, Traditional Medicine'
        }
        
        # Try to extract chapter/section information
        chapter_matches = re.findall(r'CHAPTER\s+[IVX]+[.\s]*([^\n]+)', text, re.IGNORECASE)
        if chapter_matches:
            metadata['chapters'] = chapter_matches
        
        return metadata
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Complete document processing pipeline.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing processed chunks and metadata
        """
        # Load the document
        raw_text = self.load_document(file_path)
        
        # Clean the text
        cleaned_text = self.clean_text(raw_text)
        
        # Extract metadata
        metadata = self.extract_metadata(cleaned_text)
        
        # Create chunks
        chunks = self.chunk_text(cleaned_text)
        
        # Add metadata to each chunk
        for chunk in chunks:
            chunk.update({
                'source_file': os.path.basename(file_path),
                'title': metadata['title'],
                'author': metadata['author'],
                'subject': metadata['subject']
            })
        
        return {
            'chunks': chunks,
            'metadata': metadata,
            'cleaned_text': cleaned_text
        }

    def get_chunk_preview(self, chunks: List[Dict[str, Any]], limit: int = 5) -> str:
        """
        Get a preview of the first few chunks for debugging.
        
        Args:
            chunks: List of chunk dictionaries
            limit: Number of chunks to preview
            
        Returns:
            Preview string
        """
        preview = f"Document processed into {len(chunks)} chunks:\n\n"
        
        for i, chunk in enumerate(chunks[:limit]):
            preview += f"Chunk {i+1} (ID: {chunk['id']}, Length: {chunk['length']}):\n"
            preview += f"{chunk['content'][:200]}...\n\n"
            
        if len(chunks) > limit:
            preview += f"... and {len(chunks) - limit} more chunks"
            
        return preview