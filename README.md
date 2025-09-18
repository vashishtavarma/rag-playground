# RAG Playground - Interactive RAG System Demo

A Django web application demonstrating a full Retrieval Augmented Generation (RAG) pipeline with an interactive, step-by-step interface and real-time chat. This project provides a hands-on demo of modern RAG workflows, including document upload, chunking, embedding, vector storage, retrieval, augmentation, and LLM-based response generation.

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

- **Django**: Web framework, ORM, and admin
- **FastEmbed**: Text embedding (BAAI/bge-small-en)
- **ChromaDB**: Local vector database for document embeddings
- **Google Gemini API**: LLM for response generation (optional)
- **PyPDF2**: PDF parsing
- **NumPy**: Similarity calculations
- **scikit-learn**: Similarity metrics
- **HTML/CSS/JavaScript**: Interactive frontend

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

## 📦 Project Structure

```bash
rag-playground/
├── rag/                    # Django project directory
│   ├── manage.py           # Django management script
│   ├── db.sqlite3          # SQLite database file
│   ├── ragAPP/             # Main RAG application
│   │   ├── __init__.py
│   │   ├── admin.py
│   │   ├── apps.py
│   │   ├── models.py       # Document, ChatSession, etc.
│   │   ├── tests.py        # Django tests
│   │   ├── urls.py         # App URL routing
│   │   ├── vector_db.py    # ChromaDB vector storage logic
│   │   ├── views.py        # All pipeline and chat logic
│   │   └── templatetags/   # Custom template tags (if any)
├── chroma_storage/         # Vector DB persistent storage (ignored by git)
├── test/                   # Sample test files (TXT, PDF, DOCX)
├── venv/                   # Python virtual environment
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore patterns
├── .env                    # Environment variables (optional)
├── LICENSE                 # MIT License
└── README.md               # Project documentation
```

## 📁 API Endpoints

- `/input-parsing/`: Upload/parse files
- `/chunking/`: Text chunking
- `/vector-embedding/`: Embedding generation
- `/vector-storage/`: Store embeddings
- `/retrieval/`: Retrieve relevant chunks
- `/augmentation/`: Prompt construction
- `/generation/`: LLM response generation
- `/knowledge-base/`: Manage documents
- `/chat-interface/`: Chat UI
- `/chat-query/`: AJAX chat endpoint

*See `rag/ragAPP/urls.py` for the full list of endpoints.*

## ⚠️ Limitations

- Only TXT and PDF files are supported for upload/processing
- DOCX files are not supported
- Requires a Google API key for full LLM functionality (otherwise uses template responses)
- Single-user application (no authentication/multi-user support)
- Basic chunking (character, sentence, or recursive; no advanced semantic chunking by default)
- No hybrid search or advanced re-ranking
- Intended for demonstration and educational purposes

## 🔧 Environment Variables

Optional environment variables can be set in a `.env` file:
```bash
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
