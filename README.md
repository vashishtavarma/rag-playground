# RAG System - Retrieval Augmented Generation Demo

A complete Django web application that demonstrates the full RAG (Retrieval Augmented Generation) pipeline through an interactive step-by-step interface.

## 🚀 Features

This application provides a hands-on demonstration of all 7 key steps in the RAG pipeline:

1. **Input Parsing** - Upload and parse TXT/PDF files
2. **Text Chunking** - Break documents into manageable chunks with configurable size and overlap
3. **Vector Embedding** - Generate embeddings using FastEmbed (BAAI/bge-small-en model)
4. **Vector Storage** - Store embeddings in a mock vector database
5. **Retrieval** - Perform similarity search to find relevant chunks
6. **Augmentation** - Construct prompts with retrieved context
7. **Generation** - Simulate LLM response generation

## 🛠️ Technologies Used

- **Django** - Web framework
- **FastEmbed** - Text embedding generation
- **PyPDF2** - PDF file processing
- **HTML/CSS/JavaScript** - Frontend interface

## 📋 Prerequisites

- Python 3.8+
- pip package manager

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag-system
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

4. **Apply database migrations**
   ```bash
   cd rag
   python manage.py migrate
   ```

5. **Run the development server**
   ```bash
   python manage.py runserver
   ```

6. **Access the application**
   Open your browser and navigate to: `http://127.0.0.1:8000`

## 🎯 Usage

### Step-by-Step RAG Pipeline

The application provides an intuitive navigation interface with cards for each step:

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
- Mock vector database storage simulation
- Bulk operations with session-based persistence

#### 5. Retrieval
- Demonstrate similarity search functionality
- Mock query processing with random similarity scores
- Show top-K retrieval results

#### 6. Augmentation
- Construct prompts with retrieved context
- Demonstrate how context is integrated with user queries
- Preview the final prompt structure

#### 7. Generation
- Simulate LLM response generation
- Show complete RAG pipeline output
- Demonstrate grounded response generation

### Input Methods

The application supports multiple input methods:
- **File Upload**: TXT and PDF files
- **Direct Text**: Paste text directly into forms
- **Line Input**: Process text line by line

## 📁 Project Structure

```
rag-pipeline/
├── rag/                    # Django project directory
│   ├── rag/               # Project configuration
│   │   ├── settings.py
│   │   ├── urls.py
│   │   └── wsgi.py
│   ├── ragAPP/            # Main application
│   │   ├── views.py       # RAG pipeline logic
│   │   ├── urls.py        # URL routing
│   │   └── templates/     # HTML templates
│   ├── db.sqlite3         # Database file
│   └── manage.py          # Django management script
├── test/                  # Sample test files
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🧪 Testing

Sample test files are provided in the `test/` directory:
- `one.txt` - Plain text file
- `two.docx` - Word document (note: DOCX not currently supported)
- `three.pdf` - PDF document
- `four.txt` - Additional text file

## ⚙️ Configuration

### Embedding Model
The application uses the `BAAI/bge-small-en` model from FastEmbed, which provides:
- 384-dimensional embeddings
- Optimized for English text
- Fast inference suitable for demos

### Default Parameters
- **Chunk Size**: 200 characters
- **Overlap Size**: 50 characters
- **Embedding Dimensions**: 384
- **Top-K Retrieval**: 3 results

## 🔍 Key Components

### Text Processing Functions
- `clean_text()` - Normalize and clean input text
- `parse_basic()` - Advanced text cleaning with regex processing

### RAG Pipeline Views
- `input_parsing()` - File upload and text extraction
- `chunking()` - Text segmentation with overlap
- `vector_embedding()` - FastEmbed integration
- `vector_storage()` - Mock database operations
- `retrieval()` - Similarity search simulation
- `augmentation()` - Prompt construction
- `generation()` - Response generation demo

## 🚨 Limitations

This is a **demonstration application** with some intentional limitations:
- Vector storage is session-based (not persistent)
- Similarity search uses random scores (not actual cosine similarity)
- LLM generation is simulated (not connected to real LLM APIs)
- Limited file format support (TXT and PDF only)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- FastEmbed library for efficient text embeddings
- Django framework for rapid web development
- BAAI for the BGE embedding model
