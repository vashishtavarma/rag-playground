from django.shortcuts import render, get_object_or_404
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views.decorators.http import require_http_methods
import io
import PyPDF2
import json
import uuid
import numpy as np
from .models import Document, TextChunk, ChatSession, ChatMessage
import re





def clean_text(text: str) -> str:
    """Lowercase and collapse multiple spaces/newlines."""
    # Lowercase and split to handle any whitespace, then join with single spaces
    return ' '.join(text.lower().split())

def parse_basic(text: str) -> str:
    """Basic parsing with cleaning logic - extract and cleanup text"""
    import re
    
    # Step 1: Normalize to UTF-8 (already handled by file reading)
    cleaned_text = text
    
    # Step 2: Remove junk characters
    cleaned_text = cleaned_text.replace('\x0c', '')  # Remove form feed
    cleaned_text = cleaned_text.replace('\r', '')    # Remove carriage returns
    
    # Step 3: Fix broken words (hyphenated line breaks)
    cleaned_text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', cleaned_text)
    
    # Step 4: Replace multiple spaces with single space
    cleaned_text = re.sub(r' {2,}', ' ', cleaned_text)
    
    # Step 5: Remove extra newlines (keep max 2 for paragraph separation)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    
    # Step 6: Strip leading/trailing whitespace from each line
    lines = cleaned_text.split('\n')
    cleaned_lines = [line.strip() for line in lines]
    cleaned_text = '\n'.join(cleaned_lines)
    
    # Step 7: Final cleanup - remove leading/trailing whitespace from entire text
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text




def home(request: HttpRequest):
    """Home page with RAG pipeline navigation cards"""
    return render(request, "ragAPP/home.html")


def input_parsing(request: HttpRequest):
    """Step 1: Input Parsing - Upload and parse TXT/PDF files"""
    raw_text = None
    parsed_result = None

    if request.method == "POST" and request.FILES.get("file"):
        uploaded_file = request.FILES["file"]
        filename = uploaded_file.name.lower()
        
        # Read raw text depending on file type
        if filename.endswith(".txt"):
            raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
        elif filename.endswith(".pdf") and PyPDF2 is not None:
            # Read PDF pages
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            pages_text = [page.extract_text() or "" for page in pdf_reader.pages]
            raw_text = "\n".join(pages_text)
        else:
            raw_text = "Unsupported file format or missing dependency."

        # Apply basic parsing if we have raw text
        if raw_text:
            parsed_result = parse_basic(raw_text)
    
    # Calculate character reduction for basic parsing
    char_reduction = 0
    if raw_text and parsed_result:
        char_reduction = len(raw_text) - len(parsed_result)
    
    context = {
        "raw_text": raw_text,
        "parsed_result": parsed_result,
        "char_reduction": char_reduction,
    }
    return render(request, "ragAPP/input_parsing.html", context)


def chunking(request: HttpRequest):
    """Step 2: Text Chunking"""
    chunks = None
    original_length = 0
    total_chunks = 0
    # Default to sentence-based chunking: 5 sentences per chunk, 2 sentence overlap
    chunk_size = 5
    overlap_size = 2
    input_method = "file"
    
    if request.method == "POST":
        input_method = request.POST.get("input_method", "file")
        # Get parameters
        chunk_size = int(request.POST.get("chunk_size", 5))
        overlap_size = int(request.POST.get("overlap_size", 2))

        # Enforce sensible bounds: chunk_size must be > 2 sentences
        if chunk_size <= 2:
            chunk_size = 3
        # Overlap cannot be negative or >= chunk_size
        if overlap_size < 0:
            overlap_size = 0
        if overlap_size >= chunk_size:
            overlap_size = max(0, chunk_size - 1)
        
        # Get text input
        raw_text = ""
        if request.FILES.get("file"):
            uploaded_file = request.FILES["file"]
            filename = uploaded_file.name.lower()
            if filename.endswith(".txt"):
                raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
            elif filename.endswith(".pdf") and PyPDF2 is not None:
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                pages_text = [page.extract_text() or "" for page in pdf_reader.pages]
                raw_text = "\n".join(pages_text)
        elif request.POST.get("text_input"):
            raw_text = request.POST.get("text_input")
        
        if raw_text:
            # Clean the text
            cleaned_text = clean_text(raw_text)
            original_length = len(cleaned_text)

            # Split into sentences using basic punctuation boundaries
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', cleaned_text) if s.strip()]

            # Create sentence-based chunks with overlap
            chunks = []
            start = 0
            while start < len(sentences):
                end = start + chunk_size
                chunk_sentences = sentences[start:end]
                if not chunk_sentences:
                    break
                chunk_text = " ".join(chunk_sentences).strip()
                if chunk_text:
                    chunks.append(chunk_text)

                # Move start index considering sentence overlap
                start = end - overlap_size
                if start >= len(sentences):
                    break
            
            total_chunks = len(chunks)
    
    context = {
        "chunks": chunks,
        "original_length": original_length,
        "total_chunks": total_chunks,
        "chunk_size": chunk_size,
        "overlap_size": overlap_size,
        "input_method": input_method,
    }
    return render(request, "ragAPP/chunking.html", context)


def vector_embedding(request: HttpRequest):
    """Step 3: Vector Embedding using FastEmbed"""
    from fastembed import TextEmbedding
    
    embeddings = None
    total_chunks = 0
    vector_dimensions = 384  # Fixed dimension for BAAI/bge-small-en
    model_name = "BAAI/bge-small-en"  # Fixed model
    input_method = "file"
    
    if request.method == "POST":
        input_method = request.POST.get("input_method", "file")
        raw_text = ""
        chunks = []
        
        if request.FILES.get("file"):
            uploaded_file = request.FILES["file"]
            filename = uploaded_file.name.lower()
            if filename.endswith(".txt"):
                raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
            elif filename.endswith(".pdf") and PyPDF2 is not None:
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                pages_text = [page.extract_text() or "" for page in pdf_reader.pages]
                raw_text = "\n".join(pages_text)
            
            if raw_text:
                # Clean and chunk the text (5-sentence chunks, 1-sentence overlap)
                cleaned_text = clean_text(raw_text)
                sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', cleaned_text) if s.strip()]
                sentence_chunk_size = 5
                sentence_overlap = 1
                start = 0
                while start < len(sentences):
                    end = start + sentence_chunk_size
                    chunk_sentences = sentences[start:end]
                    if not chunk_sentences:
                        break
                    chunk_text = " ".join(chunk_sentences).strip()
                    if chunk_text:
                        chunks.append(chunk_text)
                    start = end - sentence_overlap
                    if start >= len(sentences):
                        break
                        
        elif request.POST.get("text_input"):
            raw_text = request.POST.get("text_input")
            cleaned_text = clean_text(raw_text)
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', cleaned_text) if s.strip()]
            sentence_chunk_size = 5
            sentence_overlap = 1
            start = 0
            while start < len(sentences):
                end = start + sentence_chunk_size
                chunk_sentences = sentences[start:end]
                if not chunk_sentences:
                    break
                chunk_text = " ".join(chunk_sentences).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                start = end - sentence_overlap
                if start >= len(sentences):
                    break
                    
        elif request.POST.get("lines_input"):
            lines_text = request.POST.get("lines_input")
            # Each line becomes a chunk
            lines = lines_text.split('\n')
            for line in lines:
                cleaned_line = clean_text(line)
                if cleaned_line.strip():
                    chunks.append(cleaned_line.strip())
        
        if chunks:
            print(f"🚀 Starting embedding process for {len(chunks)} chunks...")
            
            # Initialize the FastEmbed model
            print(f"📦 Loading FastEmbed model: {model_name}")
            model = TextEmbedding(model_name=model_name)
            print("✅ Model loaded successfully!")
            
            embeddings = []
            total_chunks = len(chunks)
            
            # Generate embeddings for all chunks at once (FastEmbed is optimized for batch processing)
            print(f"🔄 Processing {total_chunks} chunks...")
            vectors = list(model.embed(chunks))
            
            for i, (chunk, vector) in enumerate(zip(chunks, vectors), 1):
                # print(f"  📝 Processing chunk {i}/{total_chunks}: '{chunk[:50]}{'...' if len(chunk) > 50 else ''}'")
                
                # Convert numpy array to list and round to 6 decimal places
                vector_list = [round(float(x), 6) for x in vector]
                
                embeddings.append({
                    'text': chunk,
                    'vector': vector_list
                })
                
                # print(f"  ✅ Chunk {i} embedded successfully ({len(vector_list)} dimensions)")
            
            # Update vector_dimensions based on actual embedding size
            if embeddings:
                vector_dimensions = len(embeddings[0]['vector'])
            
            print(f"🎉 All {total_chunks} chunks processed successfully!")
    
    context = {
        "embeddings": embeddings,
        "total_chunks": total_chunks,
        "vector_dimensions": vector_dimensions,
        "model_name": model_name,
        "input_method": input_method,
    }
    return render(request, "ragAPP/vector_embedding.html", context)


def vector_storage(request: HttpRequest):
    """Step 4: Vector Storage - Complete Pipeline Demo"""
    import random
    import PyPDF2
    
    # Initialize session storage for vectors if not exists
    if 'vector_storage' not in request.session:
        request.session['vector_storage'] = []
    
    stored_vectors = request.session['vector_storage']
    message = None
    error = None
    input_method = "file"
    
    if request.method == "POST":
        action = request.POST.get("action")
        
        if action == "process_text":
            input_method = request.POST.get("input_method", "file")
            raw_text = ""
            
            # Step 1: Input Parsing
            if request.FILES.get("file"):
                uploaded_file = request.FILES["file"]
                filename = uploaded_file.name.lower()
                if filename.endswith(".txt"):
                    raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
                elif filename.endswith(".pdf") and PyPDF2 is not None:
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    pages_text = [page.extract_text() or "" for page in pdf_reader.pages]
                    raw_text = "\n".join(pages_text)
                else:
                    error = "❌ Unsupported file format. Please upload TXT or PDF files."
            elif request.POST.get("text_input"):
                raw_text = request.POST.get("text_input")
            else:
                error = "❌ Please provide either a file or text input"
            
            if raw_text and not error:
                # Step 2: Text Cleaning
                cleaned_text = clean_text(raw_text)
                
                # Step 3: Sentence-based chunking (5 sentences, 1 overlap)
                chunks = []
                sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', cleaned_text) if s.strip()]
                sentence_chunk_size = 5
                sentence_overlap = 1
                start = 0
                while start < len(sentences):
                    end = start + sentence_chunk_size
                    chunk_sentences = sentences[start:end]
                    if not chunk_sentences:
                        break
                    chunk_text = " ".join(chunk_sentences).strip()
                    if chunk_text:
                        chunks.append(chunk_text)
                    start = end - sentence_overlap
                    if start >= len(sentences):
                        break
                
                if chunks:
                    print(f"🚀 Processing {len(chunks)} chunks for vector storage...")
                    
                    # Step 4: Generate FastEmbed embeddings and store
                    try:
                        from fastembed import TextEmbedding
                        
                        print("📦 Loading FastEmbed model: BAAI/bge-small-en")
                        model = TextEmbedding(model_name="BAAI/bge-small-en")
                        print("✅ FastEmbed model loaded successfully!")
                        
                        print(f"🧠 Generating embeddings for {len(chunks)} chunks...")
                        vectors = list(model.embed(chunks))
                        print(f"✅ Generated {len(vectors)} embedding vectors")
                        
                        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                            # Convert numpy array to list and round to 6 decimal places
                            vector_list = [round(float(x), 6) for x in vector]
                            vector_id = len(stored_vectors) + 1
                            
                            stored_vectors.append({
                                'id': vector_id,
                                'text': chunk,
                                'embedding': vector_list,
                                'dimensions': len(vector_list)
                            })
                            print(f"  ✅ Chunk {i+1}/{len(chunks)} embedded - ID: {vector_id}, Dims: {len(vector_list)}")
                        
                        request.session['vector_storage'] = stored_vectors
                        request.session.modified = True
                        
                        message = f"✅ Successfully processed and stored {len(chunks)} text chunks with FastEmbed embeddings ({len(vectors[0])}D each)"
                        print(f"🎉 Stored {len(chunks)} vectors with real embeddings in session storage!")
                        
                    except Exception as e:
                        error = f"❌ Error generating embeddings: {str(e)}"
                        print(f"❌ EMBEDDING ERROR: {str(e)}")
                else:
                    error = "❌ No valid chunks were generated from the input text"
                    
        elif action == "clear_all":
            # Clear all stored vectors
            request.session['vector_storage'] = []
            stored_vectors = []
            message = "✅ All vectors cleared from storage"
    
    context = {
        "stored_vectors": stored_vectors,
        "total_vectors": len(stored_vectors),
        "message": message,
        "error": error,
        "input_method": input_method,
    }
    return render(request, "ragAPP/vector_storage.html", context)


def retrieval(request: HttpRequest):
    """Step 5: Retrieval - Query Processing and Similarity Search Demo"""
    import random
    import math
    
    # Get stored vectors from session
    stored_vectors = request.session.get('vector_storage', [])
    
    # Demo query and processing
    demo_query = "What is RAG?"
    query_embedding = [round(random.uniform(-1.0, 1.0), 6) for _ in range(384)]
    
    # Simulate similarity search results
    retrieved_chunks = []
    if stored_vectors:
        # Simulate finding top 3 most similar chunks
        # In reality, this would use cosine similarity calculation
        num_results = min(3, len(stored_vectors))
        selected_indices = random.sample(range(len(stored_vectors)), num_results)
        
        for i, idx in enumerate(selected_indices):
            vector = stored_vectors[idx]
            # Simulate similarity scores (higher = more similar)
            similarity_score = round(random.uniform(0.7, 0.95), 3)
            
            retrieved_chunks.append({
                'rank': i + 1,
                'chunk_id': vector['id'],
                'text': vector['text'],
                'similarity_score': similarity_score,
                'embedding': vector['embedding'][:10]  # Show first 10 dimensions only
            })
        
        # Sort by similarity score (highest first)
        retrieved_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    context = {
        "demo_query": demo_query,
        "query_embedding": query_embedding[:10],  # Show first 10 dimensions
        "full_query_embedding": query_embedding,
        "total_stored_vectors": len(stored_vectors),
        "retrieved_chunks": retrieved_chunks,
        "has_stored_vectors": len(stored_vectors) > 0,
    }
    return render(request, "ragAPP/retrieval.html", context)


def augmentation(request: HttpRequest):
    """Step 6: Augmentation - Prompt Construction Demo"""
    import random
    
    # Get stored vectors from session
    stored_vectors = request.session.get('vector_storage', [])
    
    # Demo query (same as retrieval step)
    demo_query = "What is RAG?"
    
    # Simulate retrieved chunks (top 3 most relevant)
    retrieved_chunks = []
    if stored_vectors:
        # Simulate the same retrieval results as Step 5
        num_results = min(3, len(stored_vectors))
        selected_indices = random.sample(range(len(stored_vectors)), num_results)
        
        for i, idx in enumerate(selected_indices):
            vector = stored_vectors[idx]
            similarity_score = round(random.uniform(0.7, 0.95), 3)
            
            retrieved_chunks.append({
                'rank': i + 1,
                'chunk_id': vector['id'],
                'text': vector['text'],
                'similarity_score': similarity_score
            })
        
        # Sort by similarity score (highest first)
        retrieved_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    # Create augmented prompt
    augmented_prompt = f"""User Question: {demo_query}

Context:
"""
    
    for i, chunk in enumerate(retrieved_chunks, 1):
        augmented_prompt += f"{i}. {chunk['text']}\n"
    
    augmented_prompt += f"""
Please answer the user's question using the provided context. If the context doesn't contain relevant information, say so clearly."""
    
    context = {
        "demo_query": demo_query,
        "retrieved_chunks": retrieved_chunks,
        "augmented_prompt": augmented_prompt,
        "total_stored_vectors": len(stored_vectors),
        "has_stored_vectors": len(stored_vectors) > 0,
    }
    return render(request, "ragAPP/augmentation.html", context)


def generation(request: HttpRequest):
    """Step 7: Generation - LLM Response Generation Demo"""
    import random
    
    # Get stored vectors from session
    stored_vectors = request.session.get('vector_storage', [])
    
    # Demo query (consistent with previous steps)
    demo_query = "What is RAG?"
    
    # Simulate retrieved chunks (same as augmentation step)
    retrieved_chunks = []
    if stored_vectors:
        num_results = min(3, len(stored_vectors))
        selected_indices = random.sample(range(len(stored_vectors)), num_results)
        
        for i, idx in enumerate(selected_indices):
            vector = stored_vectors[idx]
            similarity_score = round(random.uniform(0.7, 0.95), 3)
            
            retrieved_chunks.append({
                'rank': i + 1,
                'chunk_id': vector['id'],
                'text': vector['text'],
                'similarity_score': similarity_score
            })
        
        retrieved_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    # Create augmented prompt (same as augmentation step)
    augmented_prompt = f"""User Question: {demo_query}

Context:
"""
    
    for i, chunk in enumerate(retrieved_chunks, 1):
        augmented_prompt += f"{i}. {chunk['text']}\n"
    
    augmented_prompt += f"""
Please answer the user's question using the provided context. If the context doesn't contain relevant information, say so clearly."""
    
    # Generate a realistic LLM response
    llm_response = """RAG (Retrieval Augmented Generation) is a powerful AI technique that combines the strengths of information retrieval systems with large language models. 

Based on the provided context, RAG works by first retrieving relevant information from a knowledge base or document collection, then using that retrieved context to augment the language model's generation process. This approach helps ensure that the generated responses are more accurate, factual, and grounded in specific source material.

The key advantage of RAG is that it allows language models to access and utilize external knowledge beyond what was learned during training, making responses more reliable and up-to-date. This is particularly valuable for applications requiring factual accuracy or domain-specific knowledge."""
    
    context = {
        "demo_query": demo_query,
        "retrieved_chunks": retrieved_chunks,
        "augmented_prompt": augmented_prompt,
        "llm_response": llm_response,
        "total_stored_vectors": len(stored_vectors),
        "has_stored_vectors": len(stored_vectors) > 0,
    }
    return render(request, "ragAPP/generation.html", context)


def document_text_interface(request: HttpRequest):
    """Card-based interface for document upload and text processing"""
    return render(request, "ragAPP/document_text_interface.html")


def knowledge_base(request: HttpRequest):
    """Knowledge Base - Document upload and management interface"""
    print("\n" + "="*60)
    print("📋 KNOWLEDGE BASE VIEW - Starting document management")
    print("="*60)
    
    documents = Document.objects.all().order_by('-uploaded_at')
    print(f"📊 Found {documents.count()} existing documents in database")
    
    message = None
    error = None
    
    if request.method == "POST":
        action = request.POST.get("action")
        print(f"🎯 POST request received with action: {action}")
        
        if action == "upload_document":
            print("\n📤 DOCUMENT UPLOAD PROCESS STARTED")
            print("-" * 40)
            
            title = request.POST.get("title", "")
            uploaded_file = request.FILES.get("file")
            
            print(f"📝 Document title: '{title}'")
            print(f"📁 File uploaded: {uploaded_file.name if uploaded_file else 'None'}")
            
            if not title:
                error = "❌ Please provide a document title"
                print("❌ ERROR: No title provided")
            elif not uploaded_file:
                error = "❌ Please select a file to upload"
                print("❌ ERROR: No file uploaded")
            else:
                filename = uploaded_file.name.lower()
                print(f"🔍 Processing file: {filename}")
                
                # Process file content
                raw_text = ""
                file_type = ""
                
                if filename.endswith(".txt"):
                    print("📄 Detected TXT file - reading content...")
                    raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
                    file_type = "txt"
                    print(f"✅ TXT file read successfully - {len(raw_text)} characters")
                elif filename.endswith(".pdf"):
                    print("📕 Detected PDF file - extracting text...")
                    try:
                        pdf_reader = PyPDF2.PdfReader(uploaded_file)
                        pages_text = [page.extract_text() or "" for page in pdf_reader.pages]
                        raw_text = "\n".join(pages_text)
                        file_type = "pdf"
                        print(f"✅ PDF processed successfully - {len(pdf_reader.pages)} pages, {len(raw_text)} characters")
                    except Exception as e:
                        error = f"❌ Error reading PDF: {str(e)}"
                        print(f"❌ PDF ERROR: {str(e)}")
                else:
                    error = "❌ Unsupported file format. Please upload TXT or PDF files."
                    print(f"❌ ERROR: Unsupported file format: {filename}")
                
                if raw_text and not error:
                    print(f"\n🧹 CLEANING TEXT - Original length: {len(raw_text)} characters")
                    # Clean the text
                    cleaned_text = parse_basic(raw_text)
                    print(f"✅ Text cleaned - New length: {len(cleaned_text)} characters")
                    
                    print(f"\n💾 CREATING DOCUMENT RECORD")
                    # Create document
                    document = Document.objects.create(
                        title=title,
                        file_type=file_type,
                        content=cleaned_text
                    )
                    print(f"✅ Document created with ID: {document.id}")
                    
                    # Process document into chunks with embeddings
                    print(f"\n🔄 PROCESSING DOCUMENT INTO CHUNKS")
                    try:
                        process_document_chunks(document)
                        document.processed = True
                        document.save()
                        print(f"✅ Document processing completed successfully!")
                        message = f"✅ Document '{title}' uploaded and processed successfully!"
                    except Exception as e:
                        error = f"❌ Error processing document: {str(e)}"
                        print(f"❌ PROCESSING ERROR: {str(e)}")
                        document.delete()  # Clean up if processing failed
                        print("🗑️ Document record deleted due to processing failure")
        
        elif action == "delete_document":
            doc_id = request.POST.get("document_id")
            try:
                document = Document.objects.get(id=doc_id)
                document.delete()
                message = f"✅ Document deleted successfully!"
            except Document.DoesNotExist:
                error = "❌ Document not found"
    
    context = {
        "documents": documents,
        "message": message,
        "error": error,
    }
    return render(request, "ragAPP/knowledge_base.html", context)


def process_document_chunks(document):
    """Process document into chunks with embeddings using FastEmbed"""
    print(f"\n🔧 CHUNK PROCESSING - Document: '{document.title}'")
    print("-" * 50)
    
    from fastembed import TextEmbedding
    
    print("📦 Loading FastEmbed model: BAAI/bge-small-en")
    # Initialize FastEmbed model
    model = TextEmbedding(model_name="BAAI/bge-small-en")
    print("✅ FastEmbed model loaded successfully")
    
    # Create sentence-based chunks
    chunks = []
    sentence_chunk_size = 5
    sentence_overlap = 1
    chunk_index = 0
    
    print(f"⚙️ Chunking parameters: sentences_per_chunk={sentence_chunk_size}, sentence_overlap={sentence_overlap}")
    
    cleaned_text = clean_text(document.content)
    print(f"📏 Cleaned text length: {len(cleaned_text)} characters")
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', cleaned_text) if s.strip()]
    
    print("\n✂️ CREATING TEXT CHUNKS (sentence-based)")
    start = 0
    while start < len(sentences):
        end = start + sentence_chunk_size
        chunk_sentences = sentences[start:end]
        if not chunk_sentences:
            break
        chunk_text = " ".join(chunk_sentences).strip()
        if chunk_text:
            chunks.append(chunk_text)
            print(f"  📄 Chunk {len(chunks)}: {len(chunk_text)} chars - '{chunk_text[:50]}{'...' if len(chunk_text) > 50 else ''}'")
        start = end - sentence_overlap
        if start >= len(sentences):
            break
    
    print(f"✅ Created {len(chunks)} text chunks")
    
    # Generate embeddings for all chunks
    if chunks:
        print(f"\n🧠 GENERATING EMBEDDINGS for {len(chunks)} chunks")
        vectors = list(model.embed(chunks))
        print(f"✅ Generated {len(vectors)} embedding vectors")
        
        print(f"\n💾 SAVING CHUNKS TO DATABASE")
        # Save chunks with embeddings
        for i, (chunk_text, vector) in enumerate(zip(chunks, vectors)):
            vector_list = [float(x) for x in vector]
            chunk_obj = TextChunk.objects.create(
                document=document,
                text=chunk_text,
                chunk_index=i,
                embedding=vector_list
            )
            print(f"  ✅ Saved chunk {i+1}/{len(chunks)} - ID: {chunk_obj.id}, Embedding dims: {len(vector_list)}")
        
        print(f"🎉 CHUNK PROCESSING COMPLETE - {len(chunks)} chunks saved with embeddings")
    else:
        print("⚠️ No chunks created - document may be too short or empty")


def chat_interface(request: HttpRequest):
    """Chat interface for querying the knowledge base"""
    # Get or create chat session
    session_id = request.session.get('chat_session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session['chat_session_id'] = session_id
    
    chat_session, created = ChatSession.objects.get_or_create(
        session_id=session_id
    )
    
    # Get chat history
    messages = chat_session.messages.all()
    
    # Get available documents
    documents = Document.objects.filter(processed=True)
    total_chunks = TextChunk.objects.count()
    
    context = {
        "messages": messages,
        "documents": documents,
        "total_chunks": total_chunks,
        "session_id": session_id,
    }
    return render(request, "ragAPP/chat_interface.html", context)


@csrf_exempt
@require_http_methods(["POST"])
def chat_query(request: HttpRequest):
    """Handle chat queries via AJAX"""
    try:
        data = json.loads(request.body)
        query = data.get('query', '').strip()
        session_id = data.get('session_id')
        
        if not query:
            return JsonResponse({'error': 'Query cannot be empty'}, status=400)
        
        if not session_id:
            return JsonResponse({'error': 'Session ID required'}, status=400)
        
        # Get chat session
        try:
            chat_session = ChatSession.objects.get(session_id=session_id)
        except ChatSession.DoesNotExist:
            return JsonResponse({'error': 'Invalid session'}, status=400)
        
        # Perform RAG query
        response, retrieved_chunks = perform_rag_query(query)
        
        # Save chat message
        chat_message = ChatMessage.objects.create(
            session=chat_session,
            message=query,
            response=response,
            retrieved_chunks=[chunk['id'] for chunk in retrieved_chunks]
        )
        
        return JsonResponse({
            'response': response,
            'retrieved_chunks': retrieved_chunks,
            'message_id': chat_message.id
        })
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def perform_rag_query(query):
    """Perform RAG query using FastEmbed for similarity search"""
    print(f"\n" + "="*60)
    print(f"🔍 RAG QUERY PROCESSING - Query: '{query}'")
    print("="*60)
    
    from fastembed import TextEmbedding
    
    print("📦 Loading FastEmbed model for query processing...")
    # Initialize FastEmbed model
    model = TextEmbedding(model_name="BAAI/bge-small-en")
    print("✅ FastEmbed model loaded")
    
    print(f"\n🧠 GENERATING QUERY EMBEDDING")
    # Generate query embedding
    query_embedding = list(model.embed([query]))[0]
    query_vector = np.array(query_embedding)
    print(f"✅ Query embedded - Vector dimensions: {len(query_embedding)}")
    
    # Get all chunks with embeddings
    chunks = TextChunk.objects.all()
    print(f"\n📊 KNOWLEDGE BASE STATUS")
    print(f"📄 Total chunks in database: {chunks.count()}")
    
    if not chunks.exists():
        print("⚠️ No chunks found in knowledge base")
        return "I don't have any documents in my knowledge base yet. Please upload some documents first.", []
    
    print(f"\n🔄 CALCULATING SIMILARITIES for {chunks.count()} chunks")
    # Calculate similarities
    similarities = []
    for i, chunk in enumerate(chunks, 1):
        chunk_vector = np.array(chunk.embedding)
        # Cosine similarity
        similarity = np.dot(query_vector, chunk_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(chunk_vector))
        similarities.append({
            'chunk': chunk,
            'similarity': float(similarity)
        })
        print(f"  📊 Chunk {i}: Similarity = {similarity:.4f} - '{chunk.text[:50]}{'...' if len(chunk.text) > 50 else ''}'")
    
    # Sort by similarity and get top 3
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    top_chunks = similarities[:3]
    
    print(f"\n🏆 TOP 3 MOST SIMILAR CHUNKS:")
    for i, item in enumerate(top_chunks, 1):
        chunk = item['chunk']
        similarity = item['similarity']
        print(f"  {i}. Similarity: {similarity:.4f} | Doc: '{chunk.document.title}' | Text: '{chunk.text[:80]}{'...' if len(chunk.text) > 80 else ''}'")
    
    # Prepare context for response
    context_text = ""
    retrieved_chunks = []
    
    print(f"\n📝 PREPARING CONTEXT FOR RESPONSE")
    for i, item in enumerate(top_chunks, 1):
        chunk = item['chunk']
        similarity = item['similarity']
        context_text += f"{i}. {chunk.text}\n\n"
        retrieved_chunks.append({
            'id': chunk.id,
            'text': chunk.text,
            'similarity': round(similarity, 3),
            'document_title': chunk.document.title
        })
    
    print(f"✅ Context prepared - {len(retrieved_chunks)} chunks, {len(context_text)} characters")
    
    # Generate response using the retrieved context
    print(f"\n🤖 GENERATING RESPONSE")
    response = generate_response(query, context_text)
    print(f"✅ Response generated - {len(response)} characters")
    
    print(f"\n🎉 RAG QUERY COMPLETE")
    print("="*60)
    
    return response, retrieved_chunks


def generate_response(query, context):
    """Generate response using Google Gemini API based on query and retrieved context"""
    print(f"\n💬 RESPONSE GENERATION WITH GEMINI API")
    print(f"📝 Query: '{query}'")
    print(f"📄 Context length: {len(context)} characters")
    
    if not context.strip():
        print("⚠️ No context available - returning default message")
        return "I couldn't find relevant information in the knowledge base to answer your question."
    
    try:
        import google.generativeai as genai
        import os
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv()
        
        # Configure Gemini API
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            print("⚠️ Google API key not found - falling back to template response")
            return generate_template_response(query, context)
        
        genai.configure(api_key=api_key)
        print("✅ Gemini API configured successfully")
        
        # Use Gemini 1.5 Flash (free model)
        model = genai.GenerativeModel('gemini-1.5-flash')
        print("📦 Using Gemini 1.5 Flash model")
        
        # Create prompt for the LLM
        prompt = f"""You are a helpful AI assistant that answers questions based on provided context from documents.

                Context from retrieved documents:
                {context.strip()}

                User Question: {query}

                Instructions:
                - If user greets you, respond with a friendly greeting
                - Answer the user's question using ONLY the information provided in the context above
                - Be concise and accurate
                - If the context doesn't contain enough information to fully answer the question, say so clearly
                - Don't make up information that's not in the context
                - Structure your response in a clear, readable format
                - If no context is provided, respond with a message that "Sorry, I don't have enough data to answer your question"

                Answer:"""
        
        print("🤖 Sending request to Gemini API...")
        response = model.generate_content(prompt)
        
        if response and response.text:
            print(f"✅ Gemini response received - {len(response.text)} characters")
            return response.text.strip()
        else:
            print("⚠️ Empty response from Gemini - falling back to template")
            return generate_template_response(query, context)
            
    except Exception as e:
        print(f"❌ Error with Gemini API: {str(e)}")
        print("🔄 Falling back to template response")
        return generate_template_response(query, context)


def generate_template_response(query, context):
    """Fallback template-based response generation"""
    print("📝 Generating template-based response as fallback")
    response = f"""Based on the information in my knowledge base, here's what I found regarding your question: "{query}"

        {context.strip()}

        This information was retrieved from the most relevant sections of the uploaded documents. If you need more specific details or have follow-up questions, please feel free to ask!"""
    
    print(f"✅ Template response generated - {len(response)} characters")
    return response

