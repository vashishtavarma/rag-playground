from django.shortcuts import render
from django.http import HttpRequest, HttpResponse
import io
import PyPDF2





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
    chunk_size = 200
    overlap_size = 50
    input_method = "file"
    
    if request.method == "POST":
        input_method = request.POST.get("input_method", "file")
        # Get parameters
        chunk_size = int(request.POST.get("chunk_size", 200))
        overlap_size = int(request.POST.get("overlap_size", 50))
        
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
            
            # Create chunks with overlap
            chunks = []
            start = 0
            while start < len(cleaned_text):
                end = start + chunk_size
                chunk = cleaned_text[start:end]
                chunks.append(chunk)
                
                # Move start position considering overlap
                start = end - overlap_size
                if start >= len(cleaned_text):
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
                # Clean and chunk the text
                cleaned_text = clean_text(raw_text)
                # Simple chunking (200 chars with 50 overlap)
                start = 0
                while start < len(cleaned_text):
                    end = start + 200
                    chunk = cleaned_text[start:end]
                    if chunk.strip():
                        chunks.append(chunk.strip())
                    start = end - 50
                    if start >= len(cleaned_text):
                        break
                        
        elif request.POST.get("text_input"):
            raw_text = request.POST.get("text_input")
            cleaned_text = clean_text(raw_text)
            # Simple chunking
            start = 0
            while start < len(cleaned_text):
                end = start + 200
                chunk = cleaned_text[start:end]
                if chunk.strip():
                    chunks.append(chunk.strip())
                start = end - 50
                if start >= len(cleaned_text):
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
                
                # Step 3: Chunking (200 chars with 50 overlap)
                chunks = []
                chunk_size = 200
                overlap_size = 50
                start = 0
                
                while start < len(cleaned_text):
                    end = start + chunk_size
                    chunk = cleaned_text[start:end]
                    if chunk.strip():
                        chunks.append(chunk.strip())
                    start = end - overlap_size
                    if start >= len(cleaned_text):
                        break
                
                if chunks:
                    print(f"🚀 Processing {len(chunks)} chunks for vector storage...")
                    
                    # Step 4: Generate random embeddings and store
                    for i, chunk in enumerate(chunks):
                        # Generate random embedding (384 dimensions)
                        random_embedding = [round(random.uniform(-1.0, 1.0), 6) for _ in range(384)]
                        vector_id = len(stored_vectors) + 1
                        
                        stored_vectors.append({
                            'id': vector_id,
                            'text': chunk,
                            'embedding': random_embedding,
                            'dimensions': 384
                        })
                    
                    request.session['vector_storage'] = stored_vectors
                    request.session.modified = True
                    
                    message = f"✅ Successfully processed and stored {len(chunks)} text chunks with random embeddings (384D each)"
                    print(f"🎉 Stored {len(chunks)} vectors in mock database!")
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

