from django.shortcuts import render
from django.http import HttpRequest, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import PyPDF2
import json
import uuid
from .models import Document, ChatSession, ChatMessage
from .vector_db import get_vector_db



def parse_document(uploaded_file):
    """Unified document parsing function for TXT and PDF files"""
    import re
    
    if not uploaded_file:
        return None, "No file provided"
    
    filename = uploaded_file.name.lower()
    raw_text = ""
    file_type = ""
    
    # Extract text based on file type
    try:
        if filename.endswith(".txt"):
            raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
            file_type = "txt"
        elif filename.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            pages_text = [page.extract_text() or "" for page in pdf_reader.pages]
            raw_text = "\n".join(pages_text)
            file_type = "pdf"
        else:
            return None, "Unsupported file format. Please upload TXT or PDF files."
    except Exception as e:
        return None, f"Error reading file: {str(e)}"
    
    if not raw_text.strip():
        return None, "File appears to be empty or unreadable"
    
    # Clean and parse the text
    cleaned_text = raw_text
    
    # Remove junk characters
    cleaned_text = cleaned_text.replace('\x0c', '')  # Remove form feed
    cleaned_text = cleaned_text.replace('\r', '')    # Remove carriage returns
    
    # Fix broken words (hyphenated line breaks)
    cleaned_text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', cleaned_text)
    
    # Replace multiple spaces with single space
    cleaned_text = re.sub(r' {2,}', ' ', cleaned_text)
    
    # Remove extra newlines (keep max 2 for paragraph separation)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    
    # Strip leading/trailing whitespace from each line
    lines = cleaned_text.split('\n')
    cleaned_lines = [line.strip() for line in lines]
    cleaned_text = '\n'.join(cleaned_lines)
    
    # Final cleanup
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text, file_type

def clean_text(text: str) -> str:
    """Simple text cleaning for chunking"""
    return ' '.join(text.lower().split())

def create_chunks_fixed_size(text: str, chunk_size: int = 200, overlap_size: int = 50) -> list:
    """Fixed-size chunking: Splits text into chunks of a fixed size with optional overlap"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap_size
        if start >= len(text):
            break
    
    return chunks


def create_chunks_token_based(text: str, max_tokens: int = 100, overlap_tokens: int = 20) -> list:
    """Token-based chunking: Splits text based on token count (approximated by words)"""
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + max_tokens
        chunk_words = words[start:end]
        chunk = ' '.join(chunk_words)
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap_tokens
        if start >= len(words):
            break
    
    return chunks


def create_chunks_recursive(text: str, chunk_size: int = 200, overlap_size: int = 50) -> list:
    """Recursive chunking: Hierarchical approach using prioritized separators"""
    separators = ['\n\n', '\n', '. ', '! ', '? ', '; ', ', ', ' ']
    
    def split_text_recursive(text, separators, chunk_size):
        if len(text) <= chunk_size:
            return [text] if text.strip() else []
        
        # Try each separator in order of priority
        for separator in separators:
            if separator in text:
                parts = text.split(separator)
                chunks = []
                current_chunk = ""
                
                for part in parts:
                    if len(current_chunk + separator + part) <= chunk_size:
                        if current_chunk:
                            current_chunk += separator + part
                        else:
                            current_chunk = part
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        # Recursively split large parts
                        if len(part) > chunk_size:
                            chunks.extend(split_text_recursive(part, separators[1:], chunk_size))
                        else:
                            current_chunk = part
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                return [chunk for chunk in chunks if chunk.strip()]
        
        # Fallback to character-based splitting
        return [text[i:i+chunk_size].strip() for i in range(0, len(text), chunk_size) if text[i:i+chunk_size].strip()]
    
    base_chunks = split_text_recursive(text, separators, chunk_size)
    
    # Add overlap if specified
    if overlap_size > 0 and len(base_chunks) > 1:
        overlapped_chunks = []
        for i, chunk in enumerate(base_chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                # Add overlap from previous chunk
                prev_chunk = base_chunks[i-1]
                overlap = prev_chunk[-overlap_size:] if len(prev_chunk) > overlap_size else prev_chunk
                overlapped_chunks.append(overlap + " " + chunk)
        return overlapped_chunks
    
    return base_chunks


def create_chunks_sentence_based(text: str, sentences_per_chunk: int = 3, overlap_sentences: int = 1) -> list:
    """Sentence-based chunking: Chunks based on natural sentence boundaries"""
    import re
    
    # Split into sentences using regex
    sentence_endings = r'[.!?]+'
    sentences = re.split(f'({sentence_endings})', text)
    
    # Reconstruct sentences with their endings
    reconstructed_sentences = []
    i = 0
    while i < len(sentences):
        sentence = sentences[i].strip()
        if i + 1 < len(sentences) and re.match(sentence_endings, sentences[i + 1]):
            sentence += sentences[i + 1]
            i += 2
        else:
            i += 1
        if sentence.strip():
            reconstructed_sentences.append(sentence.strip())
    
    # Group sentences into chunks
    chunks = []
    start = 0
    
    while start < len(reconstructed_sentences):
        end = start + sentences_per_chunk
        chunk_sentences = reconstructed_sentences[start:end]
        chunk = ' '.join(chunk_sentences)
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap_sentences
        if start >= len(reconstructed_sentences):
            break
    
    return chunks


def create_chunks_semantic(text: str, similarity_threshold: float = 0.7, min_chunk_size: int = 50) -> list:
    """Semantic chunking: Uses embeddings to split text based on semantic meaning"""
    try:
        from fastembed import TextEmbedding
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Split text into sentences first
        import re
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return [text] if text.strip() else []
        
        # Generate embeddings for each sentence
        model = TextEmbedding(model_name="BAAI/bge-small-en")
        embeddings = list(model.embed(sentences))
        embeddings = np.array(embeddings)
        
        # Find breakpoints based on semantic similarity
        breakpoints = [0]
        
        for i in range(1, len(sentences)):
            # Calculate similarity between current and previous sentence
            similarity = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
            
            # If similarity is below threshold, create a breakpoint
            if similarity < similarity_threshold:
                breakpoints.append(i)
        
        breakpoints.append(len(sentences))
        
        # Create chunks based on breakpoints
        chunks = []
        for i in range(len(breakpoints) - 1):
            start = breakpoints[i]
            end = breakpoints[i + 1]
            chunk_sentences = sentences[start:end]
            chunk = '. '.join(chunk_sentences)
            if len(chunk) >= min_chunk_size:
                chunks.append(chunk)
        
        return chunks if chunks else [text]
        
    except Exception as e:
        print(f"Error in semantic chunking: {e}")
        # Fallback to sentence-based chunking
        return create_chunks_sentence_based(text)


def create_chunks_hierarchical(text: str, chunk_size: int = 200, levels: int = 2) -> dict:
    """Hierarchical chunking: Creates chunks at multiple levels of granularity"""
    result = {}
    
    # Level 1: Large chunks (sections)
    large_chunk_size = chunk_size * 3
    level1_chunks = create_chunks_recursive(text, large_chunk_size, large_chunk_size // 4)
    result['level_1'] = {
        'chunks': level1_chunks,
        'description': f'Large sections (~{large_chunk_size} chars)',
        'count': len(level1_chunks)
    }
    
    # Level 2: Medium chunks (subsections)
    medium_chunk_size = chunk_size * 2
    level2_chunks = create_chunks_recursive(text, medium_chunk_size, medium_chunk_size // 4)
    result['level_2'] = {
        'chunks': level2_chunks,
        'description': f'Medium subsections (~{medium_chunk_size} chars)',
        'count': len(level2_chunks)
    }
    
    # Level 3: Small chunks (paragraphs)
    level3_chunks = create_chunks_recursive(text, chunk_size, chunk_size // 4)
    result['level_3'] = {
        'chunks': level3_chunks,
        'description': f'Small paragraphs (~{chunk_size} chars)',
        'count': len(level3_chunks)
    }
    
    if levels > 3:
        # Level 4: Sentence-based chunks
        level4_chunks = create_chunks_sentence_based(text, sentences_per_chunk=2)
        result['level_4'] = {
            'chunks': level4_chunks,
            'description': 'Sentence pairs',
            'count': len(level4_chunks)
        }
    
    return result


# Default chunking function
def create_chunks(text: str, chunk_size: int = 200, overlap_size: int = 50) -> list:
    """Default chunking function (fixed-size)"""
    return create_chunks_fixed_size(text, chunk_size, overlap_size)




def home(request: HttpRequest):
    """Home page with RAG pipeline navigation cards"""
    return render(request, "ragAPP/home.html")


def input_parsing(request: HttpRequest):
    """Step 1: Input Parsing - Upload and parse TXT/PDF files"""
    raw_text = None
    parsed_result = None
    error = None

    if request.method == "POST" and request.FILES.get("file"):
        uploaded_file = request.FILES["file"]
        
        # Get raw text first for comparison
        filename = uploaded_file.name.lower()
        if filename.endswith(".txt"):
            raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
        elif filename.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            pages_text = [page.extract_text() or "" for page in pdf_reader.pages]
            raw_text = "\n".join(pages_text)
        
        # Reset file pointer and use unified parsing function
        uploaded_file.seek(0)
        parsed_result, file_type = parse_document(uploaded_file)
        
        if parsed_result is None:
            error = file_type  # file_type contains error message
    
    # Calculate character reduction
    char_reduction = 0
    if raw_text and parsed_result:
        char_reduction = len(raw_text) - len(parsed_result)
    
    context = {
        "raw_text": raw_text,
        "parsed_result": parsed_result,
        "char_reduction": char_reduction,
        "error": error,
    }
    return render(request, "ragAPP/input_parsing.html", context)


def chunking(request: HttpRequest):
    """Step 2: Text Chunking with Multiple Strategies"""
    chunks = None
    hierarchical_chunks = None
    original_length = 0
    total_chunks = 0
    chunk_size = 200
    overlap_size = 50
    input_method = "file"
    chunking_method = "fixed_size"
    error = None
    processing_time = 0
    
    # Parameters for different chunking methods
    max_tokens = 100
    overlap_tokens = 20
    sentences_per_chunk = 3
    overlap_sentences = 1
    similarity_threshold = 0.7
    min_chunk_size = 50
    hierarchical_levels = 3
    
    if request.method == "POST":
        input_method = request.POST.get("input_method", "file")
        chunking_method = request.POST.get("chunking_method", "fixed_size")
        chunk_size = int(request.POST.get("chunk_size", 200))
        overlap_size = int(request.POST.get("overlap_size", 50))
        max_tokens = int(request.POST.get("max_tokens", 100))
        overlap_tokens = int(request.POST.get("overlap_tokens", 20))
        sentences_per_chunk = int(request.POST.get("sentences_per_chunk", 3))
        overlap_sentences = int(request.POST.get("overlap_sentences", 1))
        similarity_threshold = float(request.POST.get("similarity_threshold", 0.7))
        min_chunk_size = int(request.POST.get("min_chunk_size", 50))
        hierarchical_levels = int(request.POST.get("hierarchical_levels", 3))
        
        # Get text input
        raw_text = ""
        if request.FILES.get("file"):
            # Use unified parsing function
            parsed_text, file_type = parse_document(request.FILES["file"])
            if parsed_text is None:
                error = file_type  # Contains error message
            else:
                raw_text = parsed_text
        elif request.POST.get("text_input"):
            raw_text = request.POST.get("text_input")
        
        if raw_text and not error:
            import time
            start_time = time.time()
            
            # Don't clean text for semantic and sentence-based chunking to preserve structure
            if chunking_method in ['semantic', 'sentence_based', 'hierarchical']:
                text_to_chunk = raw_text
            else:
                text_to_chunk = clean_text(raw_text)
            
            original_length = len(text_to_chunk)
            
            try:
                # Apply the selected chunking method
                if chunking_method == "fixed_size":
                    chunks = create_chunks_fixed_size(text_to_chunk, chunk_size, overlap_size)
                elif chunking_method == "token_based":
                    chunks = create_chunks_token_based(text_to_chunk, max_tokens, overlap_tokens)
                elif chunking_method == "recursive":
                    chunks = create_chunks_recursive(text_to_chunk, chunk_size, overlap_size)
                elif chunking_method == "sentence_based":
                    chunks = create_chunks_sentence_based(text_to_chunk, sentences_per_chunk, overlap_sentences)
                elif chunking_method == "semantic":
                    chunks = create_chunks_semantic(text_to_chunk, similarity_threshold, min_chunk_size)
                elif chunking_method == "hierarchical":
                    hierarchical_chunks = create_chunks_hierarchical(text_to_chunk, chunk_size, hierarchical_levels)
                    chunks = hierarchical_chunks['level_3']['chunks']  # Default to level 3 for display
                else:
                    chunks = create_chunks_fixed_size(text_to_chunk, chunk_size, overlap_size)
                
                total_chunks = len(chunks) if chunks else 0
                processing_time = round((time.time() - start_time) * 1000, 2)  # Convert to milliseconds
                
            except Exception as e:
                error = f"Error in {chunking_method} chunking: {str(e)}"
                print(f"Chunking error: {e}")
    
    context = {
        "chunks": chunks,
        "hierarchical_chunks": hierarchical_chunks,
        "original_length": original_length,
        "total_chunks": total_chunks,
        "chunk_size": chunk_size,
        "overlap_size": overlap_size,
        "input_method": input_method,
        "chunking_method": chunking_method,
        "max_tokens": max_tokens,
        "overlap_tokens": overlap_tokens,
        "sentences_per_chunk": sentences_per_chunk,
        "overlap_sentences": overlap_sentences,
        "similarity_threshold": similarity_threshold,
        "min_chunk_size": min_chunk_size,
        "hierarchical_levels": hierarchical_levels,
        "processing_time": processing_time,
        "error": error,
    }
    return render(request, "ragAPP/chunking.html", context)


def vector_embedding(request: HttpRequest):
    """Step 3: Vector Embedding using FastEmbed"""
    from fastembed import TextEmbedding
    
    embeddings = None
    total_chunks = 0
    vector_dimensions = 384
    model_name = "BAAI/bge-small-en"
    input_method = "file"
    error = None
    
    if request.method == "POST":
        input_method = request.POST.get("input_method", "file")
        chunks = []
        
        if request.FILES.get("file"):
            parsed_text, file_type = parse_document(request.FILES["file"])
            if parsed_text is None:
                error = file_type
            else:
                cleaned_text = clean_text(parsed_text)
                chunks = create_chunks(cleaned_text)
                        
        elif request.POST.get("text_input"):
            raw_text = request.POST.get("text_input")
            cleaned_text = clean_text(raw_text)
            chunks = create_chunks(cleaned_text)
        
        if chunks and not error:
            try:
                model = TextEmbedding(model_name=model_name)
                embeddings = []
                total_chunks = len(chunks)
                vectors = list(model.embed(chunks))
                
                for chunk, vector in zip(chunks, vectors):
                    vector_list = [round(float(x), 6) for x in vector]
                    embeddings.append({
                        'text': chunk,
                        'vector': vector_list
                    })
                
                if embeddings:
                    vector_dimensions = len(embeddings[0]['vector'])
                
            except Exception as e:
                error = f"Error generating embeddings: {str(e)}"
    
    context = {
        "embeddings": embeddings,
        "total_chunks": total_chunks,
        "vector_dimensions": vector_dimensions,
        "model_name": model_name,
        "input_method": input_method,
        "error": error,
    }
    return render(request, "ragAPP/vector_embedding.html", context)


def vector_storage(request: HttpRequest):
    """Step 4: Vector Storage using ChromaDB"""
    vector_db = get_vector_db()
    
    message = None
    error = None
    input_method = "file"
    
    if request.method == "POST":
        action = request.POST.get("action")
        
        if action == "process_text":
            input_method = request.POST.get("input_method", "file")
            raw_text = ""
            document_title = "Unknown Document"
            
            # Use unified parsing
            if request.FILES.get("file"):
                uploaded_file = request.FILES["file"]
                document_title = uploaded_file.name
                parsed_text, file_type = parse_document(uploaded_file)
                if parsed_text is None:
                    error = f"❌ {file_type}"  # file_type contains error message
                else:
                    raw_text = parsed_text
            elif request.POST.get("text_input"):
                raw_text = request.POST.get("text_input")
                document_title = "Manual Text Input"
            else:
                error = "❌ Please provide either a file or text input"
            
            if raw_text and not error:
                cleaned_text = clean_text(raw_text)
                chunks = create_chunks(cleaned_text)
                
                if chunks:
                    try:
                        print(f"🚀 Processing {len(chunks)} chunks for ChromaDB storage...")
                        
                        # Add chunks to ChromaDB
                        chunk_ids = vector_db.add_document_chunks(document_title, chunks)
                        
                        message = f"✅ Successfully processed and stored {len(chunks)} text chunks in ChromaDB (384D embeddings)"
                        print(f"✅ Added {len(chunk_ids)} chunks to ChromaDB")
                        
                    except Exception as e:
                        error = f"❌ Error storing in ChromaDB: {str(e)}"
                        print(f"❌ ChromaDB Error: {str(e)}")
                else:
                    error = "❌ No valid chunks were generated from the input text"
                    
        elif action == "clear_all":
            try:
                vector_db.clear_all()
                message = "✅ All vectors cleared from ChromaDB storage"
            except Exception as e:
                error = f"❌ Error clearing ChromaDB: {str(e)}"
    
    # Get all stored documents from ChromaDB
    all_docs = vector_db.get_all_documents()
    stats = vector_db.get_collection_stats()
    
    context = {
        "stored_vectors": all_docs['documents'],
        "documents_by_title": all_docs['documents_by_title'],
        "total_vectors": all_docs['total_count'],
        "total_documents": all_docs['total_documents'],
        "collection_stats": stats,
        "message": message,
        "error": error,
        "input_method": input_method,
    }
    return render(request, "ragAPP/vector_storage.html", context)


def retrieval(request: HttpRequest):
    """Step 5: Retrieval - Query Processing and Similarity Search Demo using ChromaDB"""
    vector_db = get_vector_db()
    demo_query = "What is RAG?"
    
    # Get collection stats
    stats = vector_db.get_collection_stats()
    
    # Query ChromaDB for similar documents
    retrieved_chunks = []
    query_embedding = []
    
    if stats['total_documents'] > 0:
        try:
            # Get query results from ChromaDB
            results = vector_db.query_similar(demo_query, n_results=3)
            retrieved_chunks = results['results']
            
            # Get query embedding for display
            from fastembed import TextEmbedding
            model = TextEmbedding(model_name="BAAI/bge-small-en")
            query_vectors = list(model.embed([demo_query]))
            query_embedding = [round(float(x), 6) for x in query_vectors[0]]
        except Exception as e:
            print(f"Error in retrieval demo: {str(e)}")
    
    context = {
        "demo_query": demo_query,
        "query_embedding": query_embedding[:10] if query_embedding else [],
        "full_query_embedding": query_embedding or [],
        "total_stored_vectors": stats['total_documents'],
        "retrieved_chunks": retrieved_chunks,
        "has_stored_vectors": stats['total_documents'] > 0,
        "collection_stats": stats,
    }
    return render(request, "ragAPP/retrieval.html", context)


def augmentation(request: HttpRequest):
    """Step 6: Augmentation - Prompt Construction Demo using ChromaDB"""
    vector_db = get_vector_db()
    demo_query = "What is RAG?"
    
    # Get collection stats
    stats = vector_db.get_collection_stats()
    
    # Query ChromaDB for similar documents
    retrieved_chunks = []
    if stats['total_documents'] > 0:
        try:
            results = vector_db.query_similar(demo_query, n_results=3)
            retrieved_chunks = results['results']
        except Exception as e:
            print(f"Error in augmentation demo: {str(e)}")
    
    # Create augmented prompt
    augmented_prompt = f"""User Question: {demo_query}

Context:
"""
    
    for i, chunk in enumerate(retrieved_chunks, 1):
        augmented_prompt += f"{i}. {chunk['text']}\n"
    
    augmented_prompt += f"""\nPlease answer the user's question using the provided context. If the context doesn't contain relevant information, say so clearly."""
    
    context = {
        "demo_query": demo_query,
        "retrieved_chunks": retrieved_chunks,
        "augmented_prompt": augmented_prompt,
        "total_stored_vectors": stats['total_documents'],
        "has_stored_vectors": stats['total_documents'] > 0,
        "collection_stats": stats,
    }
    return render(request, "ragAPP/augmentation.html", context)



def generation(request: HttpRequest):
    """Step 7: Generation - LLM Response Generation Demo using ChromaDB"""
    vector_db = get_vector_db()
    demo_query = "What is RAG?"
    
    # Get collection stats
    stats = vector_db.get_collection_stats()
    
    # Query ChromaDB for similar documents
    retrieved_chunks = []
    if stats['total_documents'] > 0:
        try:
            results = vector_db.query_similar(demo_query, n_results=3)
            retrieved_chunks = results['results']
        except Exception as e:
            print(f"Error in generation demo: {str(e)}")
    
    # Create augmented prompt
    augmented_prompt = f"""User Question: {demo_query}

Context:
"""
    
    for i, chunk in enumerate(retrieved_chunks, 1):
        augmented_prompt += f"{i}. {chunk['text']}\n"
    
    augmented_prompt += f"""
Please answer the user's question using the provided context. If the context doesn't contain relevant information, say so clearly."""
    
    # Generate response using the existing generate_response function
    context_text = "\n".join([chunk['text'] for chunk in retrieved_chunks])
    llm_response = generate_response(demo_query, context_text)
    
    context = {
        "demo_query": demo_query,
        "retrieved_chunks": retrieved_chunks,
        "augmented_prompt": augmented_prompt,
        "llm_response": llm_response,
        "total_stored_vectors": stats['total_documents'],
        "has_stored_vectors": stats['total_documents'] > 0,
        "collection_stats": stats,
    }
    return render(request, "ragAPP/generation.html", context)


def document_text_interface(request: HttpRequest):
    """Enhanced document upload and text processing interface with ChromaDB integration"""
    vector_db = get_vector_db()
    
    message = None
    error = None
    processing_result = None
    
    if request.method == "POST":
        action = request.POST.get("action")
        
        if action == "upload_and_process":
            title = request.POST.get("title", "")
            uploaded_file = request.FILES.get("file")
            
            if not title:
                error = "❌ Please provide a document title"
            elif not uploaded_file:
                error = "❌ Please select a file to upload"
            else:
                try:
                    # Parse document
                    cleaned_text, file_type = parse_document(uploaded_file)
                    
                    if cleaned_text is None:
                        error = f"❌ {file_type}"
                    else:
                        # Create chunks
                        cleaned_content = clean_text(cleaned_text)
                        chunks = create_chunks(cleaned_content)
                        
                        if chunks:
                            # Store in ChromaDB
                            chunk_ids = vector_db.add_document_chunks(title, chunks)
                            
                            # Also store in Django models for compatibility
                            document = Document.objects.create(
                                title=title,
                                file_type=file_type,
                                content=cleaned_text,
                                processed=True
                            )
                            
                            processing_result = {
                                'document_id': document.id,
                                'title': title,
                                'file_type': file_type,
                                'content_length': len(cleaned_text),
                                'total_chunks': len(chunks),
                                'chunk_ids': chunk_ids[:5],  # Show first 5 IDs
                                'embedding_dimensions': 384
                            }
                            
                            message = f"✅ Document '{title}' processed and stored successfully!"
                        else:
                            error = "❌ No valid chunks were generated from the document"
                            
                except Exception as e:
                    error = f"❌ Error processing document: {str(e)}"
        
    
    # Get recent documents
    recent_documents = Document.objects.all().order_by('-uploaded_at')[:5]
    
    context = {
        "message": message,
        "error": error,
        "processing_result": processing_result,
        "recent_documents": recent_documents,
    }
    return render(request, "ragAPP/document_text_interface.html", context)


def knowledge_base(request: HttpRequest):
    """Knowledge Base - Document upload and management interface"""
    documents = Document.objects.all().order_by('-uploaded_at')
    message = None
    error = None
    
    if request.method == "POST":
        action = request.POST.get("action")
        
        if action == "upload_document":
            title = request.POST.get("title", "")
            uploaded_file = request.FILES.get("file")
            
            if not title:
                error = "❌ Please provide a document title"
            elif not uploaded_file:
                error = "❌ Please select a file to upload"
            else:
                cleaned_text, file_type = parse_document(uploaded_file)
                
                if cleaned_text is None:
                    error = f"❌ {file_type}"
                else:
                    document = Document.objects.create(
                        title=title,
                        file_type=file_type,
                        content=cleaned_text
                    )
                    
                    try:
                        process_document_chunks(document)
                        document.processed = True
                        document.save()
                        message = f"✅ Document '{title}' uploaded and processed successfully!"
                    except Exception as e:
                        error = f"❌ Error processing document: {str(e)}"
                        document.delete()
        
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
    """Process document into chunks with embeddings and store in ChromaDB"""
    vector_db = get_vector_db()
    cleaned_text = clean_text(document.content)
    chunks = create_chunks(cleaned_text)
    
    if chunks:
        chunk_ids = vector_db.add_document_chunks(document.title, chunks)
        return len(chunk_ids)
    return 0


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
    
    # Get available documents and ChromaDB stats
    documents = Document.objects.filter(processed=True)
    vector_db = get_vector_db()
    stats = vector_db.get_collection_stats()
    
    context = {
        "messages": messages,
        "documents": documents,
        "total_chunks": stats['total_documents'],
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
    """Perform RAG query using ChromaDB for similarity search"""
    vector_db = get_vector_db()
    stats = vector_db.get_collection_stats()
    
    if stats['total_documents'] == 0:
        return "I don't have any documents in my knowledge base yet. Please upload some documents first.", []
    
    try:
        results = vector_db.query_similar(query, n_results=5)
        context_text = ""
        db_retrieved_chunks = []
        
        for i, result in enumerate(results['results'], 1):
            context_text += f"{i}. {result['text']}\n\n"
            db_retrieved_chunks.append({
                'id': result['id'],
                'text': result['text'],
                'similarity': result['similarity_score'],
                'document_title': result['metadata'].get('document_title', 'Unknown'),
                'rank': result['rank']
            })
        
        response = generate_response(query, context_text)
        return response, db_retrieved_chunks
        
    except Exception as e:
        return f"Sorry, I encountered an error while searching the knowledge base: {str(e)}", []


def generate_response(query, context):
    """Generate response using Google Gemini API based on query and retrieved context"""
    if not context.strip():
        return "I couldn't find relevant information in the knowledge base to answer your question."
    
    try:
        import google.generativeai as genai
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            return generate_template_response(query, context)
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""You are a helpful AI assistant that answers questions based on provided context from documents.

Context from retrieved documents:
{context.strip()}

User Question: {query}

Instructions:
- Answer the user's question using ONLY the information provided in the context above
- Be concise and accurate
- If the context doesn't contain enough information to fully answer the question, say so clearly
- Don't make up information that's not in the context
- Structure your response in a clear, readable format

Answer:"""
        
        response = model.generate_content(prompt)
        
        if response and response.text:
            return response.text.strip()
        else:
            return generate_template_response(query, context)
            
    except Exception as e:
        return generate_template_response(query, context)


def generate_template_response(query, context):
    """Fallback template-based response generation"""
    return f"""Based on the information in my knowledge base, here's what I found regarding your question: "{query}"

{context.strip()}

This information was retrieved from the most relevant sections of the uploaded documents. If you need more specific details or have follow-up questions, please feel free to ask!"""

