#!/usr/bin/env python3
"""
Test script for Ask an Herbalist RAG application.
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag_pipeline import RAGPipeline

def test_rag_pipeline():
    """Test the RAG pipeline with a sample question"""
    print("🔬 Testing Ask an Herbalist RAG Pipeline...")
    
    try:
        # Initialize the pipeline
        print("📚 Initializing RAG pipeline...")
        rag = RAGPipeline()
        
        # Get pipeline stats
        stats = rag.get_stats()
        print(f"📊 Vector database contains {stats['vector_store']['document_count']} documents")
        
        # Test query
        test_question = "What are the benefits of chamomile?"
        print(f"❓ Testing question: {test_question}")
        
        response = rag.query(test_question)
        
        print("\n✅ Response received!")
        print(f"📝 Response length: {len(response['response'])} characters")
        print(f"📚 Sources used: {len(response['sources'])}")
        print(f"⏱️ Processing time: {response['processing_time']:.2f} seconds")
        
        if response['sources']:
            print("\n📖 Source information:")
            for i, source in enumerate(response['sources'][:2]):  # Show first 2 sources
                print(f"  Source {i+1}: {source['source_file']} (chunk {source['chunk_index']})")
        
        print(f"\n💬 Sample response (first 200 chars):")
        print(f"  {response['response'][:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_rag_pipeline()
    if success:
        print("\n🎉 All tests passed! The RAG application is working correctly.")
        print("🚀 You can now start the application with: ./run.sh")
    else:
        print("\n❌ Tests failed. Please check the error messages above.")
    
    sys.exit(0 if success else 1)