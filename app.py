import streamlit as st
import os
from typing import List, Dict, Any
import time
from dotenv import load_dotenv

# Import our custom modules
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.llm_interface import LLMInterface
from src.rag_pipeline import RAGPipeline

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Ask an Herbalist",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E8B57;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 10px;
        border-left: 4px solid #2E8B57;
        background-color: #f8f9fa;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #1976d2;
    }
    
    .assistant-message {
        background-color: #f1f8e9;
        border-left-color: #388e3c;
    }
    
    .stButton > button {
        background-color: #2E8B57;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #256d47;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .source-card {
        background-color: #fff;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .info-box {
        background-color: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_app():
    """Initialize the RAG pipeline and load the herbalism book."""
    if 'rag_pipeline' not in st.session_state:
        with st.spinner("🌿 Initializing the Herbalist AI... This may take a moment."):
            try:
                # Initialize the RAG pipeline
                st.session_state.rag_pipeline = RAGPipeline()
                
                # Load the herbalism book
                book_path = "herbalism_book.txt"
                if os.path.exists(book_path):
                    st.session_state.rag_pipeline.load_document(book_path)
                    st.success("✅ Herbalism knowledge base loaded successfully!")
                else:
                    st.error("❌ Herbalism book not found. Please ensure 'herbalism_book.txt' is in the project directory.")
                    return False
                    
            except Exception as e:
                st.error(f"❌ Error initializing the application: {str(e)}")
                return False
    
    return True

def display_chat_history():
    """Display the chat history with proper styling."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>🤔 You:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🌿 Herbalist:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
            
            # Show sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources from the knowledge base"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>Source {i}:</strong><br>
                            {source["content"][:300]}...
                            <br><small><em>Relevance score: {source["score"]:.3f}</em></small>
                        </div>
                        """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🌿 Ask an Herbalist</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Your AI companion for traditional herbal medicine knowledge</p>', unsafe_allow_html=True)
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("## About This App")
        st.markdown("""
        <div class="info-box">
            This AI assistant is trained on <strong>"Herbal Simples Approved for Modern Uses of Cure"</strong> 
            by W. T. Fernie, M.D., a comprehensive guide to traditional herbal medicine.
            
            Ask questions about:
            • Medicinal plants and herbs
            • Traditional remedies
            • Herbal preparations
            • Plant-based treatments
            • Historical uses of herbs
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("## Example Questions")
        example_questions = [
            "What herbs are good for digestive issues?",
            "How do you prepare chamomile tea for medicinal use?",
            "What are the benefits of echinacea?",
            "Which herbs help with sleep problems?",
            "How do you make herbal tinctures?"
        ]
        
        for question in example_questions:
            if st.button(question, key=f"example_{question}"):
                st.session_state.user_input = question
        
        st.markdown("---")
        st.markdown("### ⚠️ Important Disclaimer")
        st.markdown("""
        <small>
        This information is for educational purposes only. 
        Always consult with healthcare professionals before 
        using herbal remedies for medical conditions.
        </small>
        """, unsafe_allow_html=True)
    
    # Initialize the application
    if not initialize_app():
        return
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Chat history
        chat_container = st.container()
        with chat_container:
            display_chat_history()
        
        # Chat input
        user_input = st.text_input(
            "Ask your herbalism question:",
            key="user_input",
            placeholder="e.g., What herbs are good for headaches?",
            help="Type your question about herbs, remedies, or traditional medicine"
        )
        
        if st.button("🌿 Ask the Herbalist", type="primary") or user_input:
            if user_input:
                # Add user message to chat history
                if 'messages' not in st.session_state:
                    st.session_state.messages = []
                
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # Process the query
                with st.spinner("🔍 Searching herbal knowledge..."):
                    try:
                        response = st.session_state.rag_pipeline.query(user_input)
                        
                        # Add assistant response to chat history
                        assistant_message = {
                            "role": "assistant", 
                            "content": response["answer"],
                            "sources": response.get("sources", [])
                        }
                        st.session_state.messages.append(assistant_message)
                        
                        # Clear the input
                        st.session_state.user_input = ""
                        
                        # Rerun to update the display
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error processing your question: {str(e)}")
    
    with col2:
        st.markdown("### 🔬 Query Analysis")
        if st.session_state.messages:
            last_response = [msg for msg in st.session_state.messages if msg["role"] == "assistant"]
            if last_response and "sources" in last_response[-1]:
                sources = last_response[-1]["sources"]
                if sources:
                    st.metric("Sources Found", len(sources))
                    avg_score = sum(s["score"] for s in sources) / len(sources)
                    st.metric("Avg Relevance", f"{avg_score:.3f}")
                    
                    # Show source distribution
                    st.markdown("**Source Quality:**")
                    for i, source in enumerate(sources[:3], 1):
                        score_pct = int(source["score"] * 100)
                        st.progress(source["score"], f"Source {i}: {score_pct}%")
        
        st.markdown("---")
        
        # Statistics
        if st.session_state.messages:
            user_messages = [msg for msg in st.session_state.messages if msg["role"] == "user"]
            st.metric("Questions Asked", len(user_messages))
        
        # Reset conversation
        if st.button("🔄 New Conversation"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()