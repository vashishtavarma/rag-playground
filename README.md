# RAG Playground - Interactive RAG System Demo

A complete Django web application that demonstrates the full RAG (Retrieval Augmented Generation) pipeline through an interactive step-by-step interface with real-time chat functionality.

## 🚀 Features

This application provides a hands-on demonstration of all 7 key steps in the RAG pipeline:

1. **Input Parsing** - Upload and parse TXT/PDF files with advanced text cleaning
2. **Text Chunking** - Break documents into manageable chunks with configurable size and overlap
3. **Vector Embedding** - Generate 384-dimensional embeddings using FastEmbed (BAAI/bge-small-en model)
4. **Vector Storage** - Store embeddings in SQLite database with persistent storage
5. **Retrieval** - Perform cosine similarity search to find relevant chunks
6. **Augmentation** - Construct prompts with retrieved context
7. **Generation** - Generate responses using Google Gemini API

### Additional Features
- **Knowledge Base Management** - Upload, store, and manage documents with persistent storage
- **Interactive Chat Interface** - Real-time chat with RAG-powered responses
- **Session Management** - Maintain chat history and context across sessions
- **Multiple Input Methods** - Support for file upload, direct text input, and line-by-line processing

## 🛠️ Technologies Used

- **Django** - Web framework with SQLite database
- **FastEmbed** - High-performance text embedding generation
- **Google Gemini API** - Advanced language model for response generation
- **PyPDF2** - PDF file processing and text extraction
- **NumPy** - Numerical computations for similarity calculations
- **HTML/CSS/JavaScript** - Interactive frontend interface with AJAX

## 📋 Prerequisites

- Python 3.8+
- pip package manager

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/vashishtavarma/rag-playground.git
   cd rag-playground
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables (optional)**
   ```bash
   # Create .env file in the root directory
   echo "GOOGLE_API_KEY=your_google_api_key_here" > .env
   ```
   *Note: Google API key is optional. The system will fall back to template responses if not provided.*

5. **Apply database migrations**
   ```bash
   cd rag
   python manage.py migrate
   ```

6. **Run the development server**
   ```bash
   python manage.py runserver
   ```

7. **Access the application**
   Open your browser and navigate to: `http://127.0.0.1:8000`

## 🎯 Usage

### Step-by-Step RAG Pipeline

The application provides an intuitive navigation interface with cards for each step:

**Main Navigation:**
- **Home** - Overview of all RAG pipeline steps
- **Knowledge Base** - Document upload and management
- **Chat Interface** - Interactive RAG-powered chat

#### 1. Input Parsing
- Upload TXT or PDF files
- View raw text extraction
- See text cleaning and normalization results

#### 2. Text Chunking
- Configure chunk size and overlap parameters
- Process text through file upload or direct input
- Visualize how text is broken into chunks

#### 3. Vector Embedding
- Generate 384-dimensional embeddings using FastEmbed
- Support for file upload, text input, or line-by-line processing
- Real-time embedding generation with progress feedback

#### 4. Vector Storage
- Complete pipeline processing (parsing → chunking → embedding → storage)
- SQLite database storage with persistent data
- Bulk operations with real database persistence

#### 5. Retrieval
- Demonstrate cosine similarity search functionality
- Real similarity calculations using NumPy
- Show top-K retrieval results with similarity scores

#### 6. Augmentation
- Construct prompts with retrieved context
- Demonstrate how context is integrated with user queries
- Preview the final prompt structure

#### 7. Generation
- Real LLM response generation using Google Gemini API
- Show complete RAG pipeline output
- Demonstrate grounded response generation with fallback templates

### Input Methods

The application supports multiple input methods:
- **File Upload**: TXT and PDF files
- **Direct Text**: Paste text directly into forms
- **Line Input**: Process text line by line

## 📁 Project Structure

```
rag-playground/
├── rag/                    # Django project directory
│   ├── rag/               # Project configuration
│   │   ├── settings.py    # Django settings
│   │   ├── urls.py        # Main URL configuration
│   │   ├── wsgi.py        # WSGI configuration
│   │   └── asgi.py        # ASGI configuration
│   ├── ragAPP/            # Main RAG application
│   │   ├── models.py      # Database models (Document, TextChunk, ChatSession)
│   │   ├── views.py       # RAG pipeline logic and API endpoints
│   │   ├── urls.py        # Application URL routing
│   │   ├── admin.py       # Django admin configuration
│   │   └── templates/     # HTML templates for all views
│   ├── db.sqlite3         # SQLite database file
│   └── manage.py          # Django management script
├── test/                  # Sample test files (TXT, PDF, DOCX)
├── venv/                  # Virtual environment (created during setup)
├── .env                   # Environment variables (optional)
├── .gitignore            # Git ignore patterns
├── requirements.txt       # Python dependencies
├── LICENSE               # MIT License
└── README.md             # This documentation
```

## 🧪 Testing

Sample test files are provided in the `test/` directory:
- `one.txt` - Plain text file (2.2KB)
- `two.docx` - Word document (12.8KB) - *Note: DOCX not currently supported*
- `three.pdf` - PDF document (277KB)
- `four.txt` - Additional text file (909B)
- `five.txt` - Text file (3KB)
- `six.txt` - Text file (3.6KB)

### Testing the Application
1. **Upload Documents**: Use the Knowledge Base to upload sample files
2. **Test Pipeline Steps**: Navigate through each RAG step individually
3. **Chat Interface**: Ask questions about uploaded documents
4. **API Testing**: Use the chat interface to test real-time RAG responses

## ⚙️ Configuration

### Embedding Model
The application uses the `BAAI/bge-small-en` model from FastEmbed, which provides:
- 384-dimensional embeddings
- Optimized for English text
- Fast inference suitable for demos

### Default Parameters
- **Chunk Size**: 200 characters
- **Overlap Size**: 50 characters
- **Embedding Dimensions**: 384 (BAAI/bge-small-en)
- **Top-K Retrieval**: 3 results
- **Similarity Metric**: Cosine similarity
- **LLM Model**: Google Gemini 1.5 Flash

## 🔍 Key Components

### Text Processing Functions
- `clean_text()` - Normalize and clean input text
- `parse_basic()` - Advanced text cleaning with regex processing

### RAG Pipeline Views
- `input_parsing()` - File upload and text extraction with advanced cleaning
- `chunking()` - Text segmentation with configurable overlap
- `vector_embedding()` - FastEmbed integration with real-time processing
- `vector_storage()` - SQLite database operations with persistent storage
- `retrieval()` - Cosine similarity search with NumPy calculations
- `augmentation()` - Prompt construction with retrieved context
- `generation()` - Response generation using Google Gemini API

### Additional Components
- `knowledge_base()` - Document management interface
- `chat_interface()` - Real-time chat with RAG responses
- `chat_query()` - AJAX endpoint for chat processing
- `perform_rag_query()` - Complete RAG pipeline execution
- `similarity_search()` - Unified similarity search function

## 🚨 Limitations

This is a **demonstration application** with some limitations:
- Limited file format support (TXT and PDF only, DOCX not supported)
- Requires Google API key for full LLM functionality (falls back to templates)
- Single-user application (no multi-user authentication)
- Basic chunking strategy (character-based, not semantic)
- No advanced retrieval techniques (hybrid search, re-ranking)

## 🔧 Environment Variables

Optional environment variables can be set in a `.env` file:
```
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

**Note**: The application works without the API key but will use template-based responses instead of AI-generated ones.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FastEmbed** library for efficient text embeddings
- **Django** framework for rapid web development
- **BAAI** for the BGE embedding model
- **Google** for the Gemini API
- **NumPy** for numerical computations
- **PyPDF2** for PDF processing capabilities
