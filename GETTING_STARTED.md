# 🌿 Ask an Herbalist - Getting Started Guide

Welcome to your RAG LLM application for herbalism! This application has been built and is ready to use.

## ✅ What's Been Set Up

1. **Complete RAG Pipeline**: The application includes document processing, vector storage, and LLM interface
2. **Herbalism Knowledge Base**: The book "Herbal Simples Approved for Modern Uses of Cure" has been processed and stored
3. **Vector Database**: ChromaDB has been initialized with embeddings from the herbalism book
4. **Modern UI**: Streamlit interface with a sleek, herbalism-themed design
5. **Smart Features**: Conversation history, source citations, and safety warnings

## 🚀 Running the Application

### Option 1: Using the Run Script (Recommended)
```bash
./run.sh
```

### Option 2: Manual Start
```bash
# Activate virtual environment
source venv/bin/activate

# Start the application
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## 🌐 Accessing the Application

Once started, the application will be available at:
- **Local Access**: http://localhost:8501
- **Network Access**: http://[your-ip]:8501

## 🎯 How to Use

1. **Ask Questions**: Type your herbalism questions in the chat interface
2. **Get Answers**: The AI will provide responses based on the herbalism knowledge base
3. **View Sources**: See which parts of the book were used to answer your question
4. **Continue Conversations**: Ask follow-up questions to dive deeper into topics

## 💡 Example Questions to Try

- "What are the benefits of chamomile?"
- "How do I prepare a ginger remedy for digestion?"
- "What herbs are good for headaches?"
- "Tell me about the traditional uses of lavender"
- "What safety precautions should I know when using herbs?"

## 🔧 Features

- **RAG-Powered**: Uses Retrieval-Augmented Generation for accurate, source-based answers
- **Conversation Memory**: Maintains context across multiple questions
- **Source Citations**: Shows exactly which parts of the book informed each answer
- **Safety Emphasis**: Always includes appropriate medical disclaimers
- **Modern UI**: Beautiful, responsive interface optimized for herbalism discussions

## 📊 System Information

- **Vector Database**: ChromaDB with sentence-transformers embeddings
- **LLM**: Supports OpenAI API (if configured) or local fallback responses
- **Knowledge Base**: ~1,000 chunks from the herbalism reference book
- **Framework**: Streamlit for the web interface

## ⚙️ Configuration

### Adding API Keys (Optional)
If you want to use OpenAI's GPT models:

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. Restart the application

### Using Local Models
The application works without API keys by using built-in fallback responses and local embedding models.

## 🛠️ Troubleshooting

### Application Won't Start
- Make sure you're in the virtual environment: `source venv/bin/activate`
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify the vector database exists: Look for the `chroma_db` directory

### No Responses to Questions
- Check that the setup completed successfully
- Verify the herbalism book was processed (should see "Setup complete!" message)
- Try rerunning the setup: `python setup.py`

### Performance Issues
- The first query may be slower as models are loaded
- Consider using a machine with more RAM for better performance
- GPU support will improve embedding speed if available

## 🔄 Resetting the Application

To start fresh:
```bash
# Remove the vector database
rm -rf chroma_db

# Run setup again
python setup.py
```

## 📚 Technical Details

### Architecture
```
User Query → Streamlit UI → RAG Pipeline → Vector Search → LLM → Response
```

### Components
- **Document Processor**: Chunks and cleans the herbalism book
- **Vector Store**: ChromaDB for semantic search
- **LLM Interface**: Handles different model types
- **RAG Pipeline**: Orchestrates the entire process

## 🎨 Customization

The application is designed to be easily customizable:

- **UI Styling**: Modify the CSS in `app.py`
- **System Prompts**: Update prompts in `src/rag_pipeline.py`
- **Chunk Size**: Adjust in the RAG pipeline configuration
- **Models**: Change embedding or LLM models in the configuration

## 📖 Adding More Documents

To add additional herbalism books or documents:

```python
# Example: Add more documents
rag = RAGPipeline()
rag.process_documents(['new_book.txt', 'another_source.pdf'])
```

## 🤝 Support

This is a complete, working RAG application for herbalism questions. The knowledge base contains traditional herbal medicine information and emphasizes safety and professional consultation for medical conditions.

Enjoy exploring the world of traditional herbalism with your new AI assistant! 🌿