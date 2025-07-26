#!/usr/bin/env python3
"""
Setup script for Ask an Herbalist RAG application.
This script processes the herbalism book and creates the vector database.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag_pipeline import RAGPipeline

def main():
    """Main setup function"""
    print("🌿 Setting up Ask an Herbalist RAG Application...")
    
    # Initialize the RAG pipeline
    print("📚 Initializing RAG pipeline...")
    rag = RAGPipeline()
    
    # Process the herbalism book
    book_path = "herbalism_book.txt"
    if os.path.exists(book_path):
        print(f"📖 Processing herbalism book: {book_path}")
        try:
            rag.process_documents([book_path])
            print("✅ Book processed successfully!")
            print("🎯 Vector database created and populated!")
            print("\n🚀 Setup complete! You can now run the application with:")
            print("   streamlit run app.py")
        except Exception as e:
            print(f"❌ Error processing book: {e}")
            return False
    else:
        print(f"❌ Book file not found: {book_path}")
        print("Please ensure the herbalism book is downloaded.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)