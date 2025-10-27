# 🌿 Ask an Herbalist - RAG LLM Application

A modern, sleek Retrieval-Augmented Generation (RAG) application that allows users to ask questions about traditional herbalism and receive knowledgeable responses based on "Herbal Simples Approved for Modern Uses of Cure" by W. T. Fernie, M.D.

![Ask an Herbalist](https://img.shields.io/badge/AI-Herbalism-green)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange)

## 🌟 Features

- **🔍 Intelligent Question Answering**: Ask questions about herbs, remedies, and traditional medicine
- **📚 Knowledge Base**: Trained on a comprehensive historical herbalism text
- **🎨 Modern UI**: Beautiful, responsive Streamlit interface
- **🔄 Conversational**: Keep asking follow-up questions in the same session
- **📊 Source Attribution**: See exactly which parts of the knowledge base were used
- **⚡ Fast Performance**: Optimized vector search and embedding generation
- **🔧 Flexible**: Works with OpenAI API, local Ollama models, or built-in fallback
- **🛡️ Safe**: Includes appropriate medical disclaimers and safety reminders

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- 4GB+ RAM recommended
- Internet connection for downloading models (initial setup only)

### Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd ask-an-herbalist
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment (optional)**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys if using OpenAI or Ollama
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** to `http://localhost:8501`

That's it! The application will automatically:
- Download the herbalism book from Project Gutenberg
- Process and chunk the text
- Generate embeddings and build the vector database
- Start the interactive chat interface

## 📖 How to Use

### Asking Questions

Simply type your herbalism-related questions in the chat interface. For example:

- "What herbs are good for digestive issues?"
- "How do you prepare chamomile tea for medicinal use?"
- "What are the benefits of echinacea?"
- "Which herbs help with sleep problems?"
- "How do you make herbal tinctures?"

### Example Conversations

**User**: "What herbs help with headaches?"

**Herbalist**: "Based on the traditional herbalism knowledge from Dr. Fernie's text, several herbs can help with headaches:

1. **Willow Bark** - Contains salicin, similar to aspirin, and is traditionally used for pain relief
2. **Feverfew** - Specifically known for preventing and treating headaches
3. **Peppermint** - Can be applied as oil to temples or consumed as tea

The reference text mentions that willow bark should be prepared as a decoction by boiling the bark in water for 15-20 minutes..."

### Features in the Interface

- **📚 Source Viewing**: Click "Sources from the knowledge base" to see exactly what text was used
- **🔬 Query Analysis**: View relevance scores and source quality metrics
- **🔄 New Conversation**: Reset the chat to start fresh
- **💡 Example Questions**: Click on suggested questions in the sidebar

## 🏗️ Architecture

The application uses a modern RAG architecture:

```
User Question → Vector Search → Context Retrieval → LLM Generation → Response
     ↑              ↑                ↑                 ↑
Text Input    Embedding Model    ChromaDB Vector Store    GPT/Ollama/Local
```

### Components

1. **Document Processor**: Cleans and chunks the herbalism text
2. **Vector Store**: ChromaDB for semantic search with sentence transformers
3. **LLM Interface**: Supports multiple model types (OpenAI, Ollama, local)
4. **RAG Pipeline**: Orchestrates retrieval and generation
5. **Streamlit UI**: Modern, responsive chat interface

## 🔧 Configuration

### Model Options

The application supports three types of language models:

#### 1. Local Fallback (Default - No API Key Required)
- Built-in rule-based responses
- Uses knowledge from the vector database
- Perfect for privacy and offline use

#### 2. OpenAI Models (Requires API Key)
```python
# Set in .env file
OPENAI_API_KEY=your_key_here
```
Supported models: `gpt-3.5-turbo`, `gpt-4`, `gpt-4o`

#### 3. Ollama Models (Requires Local Ollama Installation)
```bash
# Install Ollama from https://ollama.ai
ollama pull llama2  # or other models
```

### Customization

You can customize the application by modifying parameters in `src/rag_pipeline.py`:

```python
rag_pipeline = RAGPipeline(
    chunk_size=1000,           # Size of text chunks
    chunk_overlap=200,         # Overlap between chunks
    embedding_model="all-MiniLM-L6-v2",  # Embedding model
    llm_type="local",          # "openai", "ollama", or "local"
    temperature=0.7,           # Response creativity
    top_k_results=5            # Number of sources to retrieve
)
```

## 📁 Project Structure

```
ask-an-herbalist/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── README.md             # This file
├── herbalism_book.txt    # Downloaded automatically
├── chroma_db/            # Vector database (created automatically)
└── src/
    ├── __init__.py
    ├── document_processor.py  # Text processing and chunking
    ├── vector_store.py       # ChromaDB interface
    ├── llm_interface.py      # Language model interfaces
    └── rag_pipeline.py       # Main RAG orchestration
```

## ⚙️ Advanced Usage

### Using with OpenAI

1. Get an API key from [OpenAI](https://platform.openai.com/api-keys)
2. Add to your `.env` file:
   ```
   OPENAI_API_KEY=sk-your-key-here
   ```
3. Restart the application

### Using with Ollama

1. Install [Ollama](https://ollama.ai)
2. Pull a model: `ollama pull llama2`
3. Update the configuration in the code or use environment variables
4. Restart the application

### Custom Knowledge

You can add your own herbalism knowledge to the database:

```python
# In the pipeline
rag_pipeline.add_custom_knowledge(
    content="Your custom herbalism text here...",
    title="My Herbalism Notes",
    author="Your Name"
)
```

## 🛡️ Safety and Disclaimers

**Important**: This application is for educational purposes only. The information provided:

- Is based on historical texts from the early 1900s
- Should not replace professional medical advice
- May not reflect current medical understanding
- Should be verified with healthcare professionals

Always consult qualified healthcare providers for medical conditions.

## 🐛 Troubleshooting

### Common Issues

1. **"No module named 'src'"**
   - Make sure you're running from the project root directory
   - Try: `python -m streamlit run app.py`

2. **Slow first startup**
   - The app downloads embedding models on first run (normal)
   - Subsequent startups will be much faster

3. **Memory issues**
   - Reduce `chunk_size` in the configuration
   - Close other applications to free memory

4. **ChromaDB errors**
   - Delete the `chroma_db` folder and restart
   - Check disk space availability

### Getting Help

1. Check the Streamlit logs for error messages
2. Ensure all dependencies are installed correctly
3. Verify the herbalism book was downloaded successfully
4. Try resetting the conversation or restarting the app

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional embedding models
- More sophisticated chunking strategies
- Enhanced UI features
- Additional knowledge sources
- Performance optimizations

## 📄 License

This project is open source and available under the MIT License.

The herbalism text "Herbal Simples Approved for Modern Uses of Cure" by W. T. Fernie, M.D. is in the public domain and provided by Project Gutenberg.

## 🙏 Acknowledgments

- **Project Gutenberg** for providing the herbalism text
- **W. T. Fernie, M.D.** for the original herbalism knowledge
- **ChromaDB** for the vector database
- **Sentence Transformers** for embedding models
- **Streamlit** for the web interface
- **OpenAI** and **Ollama** for language model support

---

**Built with ❤️ for herbalism enthusiasts and AI learners**

*Remember: This tool shares traditional knowledge for educational purposes. Always consult healthcare professionals for medical advice.*