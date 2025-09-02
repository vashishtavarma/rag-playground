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

def create_chunks(text: str, chunk_size: int = 200, overlap_size: int = 50) -> list:
    """Create text chunks with specified size and overlap"""
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
    """Step 2: Text Chunking"""
    chunks = None
    original_length = 0
    total_chunks = 0
    chunk_size = 200
    overlap_size = 50
    input_method = "file"
    error = None
    
    if request.method == "POST":
        input_method = request.POST.get("input_method", "file")
        chunk_size = int(request.POST.get("chunk_size", 200))
        overlap_size = int(request.POST.get("overlap_size", 50))
        
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
            cleaned_text = clean_text(raw_text)
            original_length = len(cleaned_text)
            chunks = create_chunks(cleaned_text, chunk_size, overlap_size)
            total_chunks = len(chunks)
    
    context = {
        "chunks": chunks,
        "original_length": original_length,
        "total_chunks": total_chunks,
        "chunk_size": chunk_size,
        "overlap_size": overlap_size,
        "input_method": input_method,
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
            # Use unified parsing function
            parsed_text, file_type = parse_document(request.FILES["file"])
            if parsed_text is None:
                error = file_type  # Contains error message
            else:
                cleaned_text = clean_text(parsed_text)
                chunks = create_chunks(cleaned_text)
                        
        elif request.POST.get("text_input"):
            raw_text = request.POST.get("text_input")
            cleaned_text = clean_text(raw_text)
            chunks = create_chunks(cleaned_text)
                    
        elif request.POST.get("lines_input"):
            lines_text = request.POST.get("lines_input")
            lines = lines_text.split('\n')
            for line in lines:
                cleaned_line = clean_text(line)
                if cleaned_line.strip():
                    chunks.append(cleaned_line.strip())
        
        if chunks and not error:
            try:
                print(f"🚀 Starting embedding process for {len(chunks)} chunks...")
                
                model = TextEmbedding(model_name=model_name)
                print("✅ Model loaded successfully!")
                
                embeddings = []
                total_chunks = len(chunks)
                vectors = list(model.embed(chunks))
                
                for i, (chunk, vector) in enumerate(zip(chunks, vectors), 1):
                    vector_list = [round(float(x), 6) for x in vector]
                    embeddings.append({
                        'text': chunk,
                        'vector': vector_list
                    })
                
                if embeddings:
                    vector_dimensions = len(embeddings[0]['vector'])
                
                print(f"🎉 All {total_chunks} chunks processed successfully!")
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
    """Step 4: Vector Storage - Complete Pipeline Demo"""
    from fastembed import TextEmbedding
    
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
            
            # Use unified parsing
            if request.FILES.get("file"):
                parsed_text, file_type = parse_document(request.FILES["file"])
                if parsed_text is None:
                    error = f"❌ {file_type}"  # file_type contains error message
                else:
                    raw_text = parsed_text
            elif request.POST.get("text_input"):
                raw_text = request.POST.get("text_input")
            else:
                error = "❌ Please provide either a file or text input"
            
            if raw_text and not error:
                cleaned_text = clean_text(raw_text)
                chunks = create_chunks(cleaned_text)
                
                if chunks:
                    try:
                        print(f"🚀 Processing {len(chunks)} chunks for vector storage...")
                        
                        model = TextEmbedding(model_name="BAAI/bge-small-en")
                        print("✅ FastEmbed model loaded successfully!")
                        
                        vectors = list(model.embed(chunks))
                        print(f"✅ Generated {len(vectors)} embedding vectors")
                        
                        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                            vector_list = [round(float(x), 6) for x in vector]
                            vector_id = len(stored_vectors) + 1
                            
                            stored_vectors.append({
                                'id': vector_id,
                                'text': chunk,
                                'embedding': vector_list,
                                'dimensions': len(vector_list)
                            })
                        
                        request.session['vector_storage'] = stored_vectors
                        request.session.modified = True
                        
                        message = f"✅ Successfully processed and stored {len(chunks)} text chunks with FastEmbed embeddings ({len(vectors[0])}D each)"
                        
                    except Exception as e:
                        error = f"❌ Error generating embeddings: {str(e)}"
                else:
                    error = "❌ No valid chunks were generated from the input text"
                    
        elif action == "clear_all":
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
    # Get stored vectors from session
    stored_vectors = request.session.get('vector_storage', [])
    demo_query = "What is RAG?"
    
    # Use unified similarity search
    retrieved_chunks = similarity_search(demo_query, stored_vectors)
    
    # Get query embedding for display
    query_embedding = []
    if stored_vectors:
        try:
            from fastembed import TextEmbedding
            model = TextEmbedding(model_name="BAAI/bge-small-en")
            query_vectors = list(model.embed([demo_query]))
            query_embedding = [round(float(x), 6) for x in query_vectors[0]]
        except Exception as e:
            print(f"Error generating query embedding: {str(e)}")
    
    context = {
        "demo_query": demo_query,
        "query_embedding": query_embedding[:10] if query_embedding else [],
        "full_query_embedding": query_embedding or [],
        "total_stored_vectors": len(stored_vectors),
        "retrieved_chunks": retrieved_chunks,
        "has_stored_vectors": len(stored_vectors) > 0,
    }
    return render(request, "ragAPP/retrieval.html", context)


def augmentation(request: HttpRequest):
    """Step 6: Augmentation - Prompt Construction Demo"""
    # Get stored vectors from session
    stored_vectors = request.session.get('vector_storage', [])
    demo_query = "What is RAG?"
    
    # Use unified similarity search
    retrieved_chunks = similarity_search(demo_query, stored_vectors)
    
    # Create augmented prompt
    augmented_prompt = f"""User Question: {demo_query}
                        Context:
                        """
    
    for i, chunk in enumerate(retrieved_chunks, 1):
        augmented_prompt += f"{i}. {chunk['text']}\n"
    
    augmented_prompt += f"""Please answer the user's question using the provided context. If the context doesn't contain relevant information, say so clearly."""
    
    context = {
        "demo_query": demo_query,
        "retrieved_chunks": retrieved_chunks,
        "augmented_prompt": augmented_prompt,
        "total_stored_vectors": len(stored_vectors),
        "has_stored_vectors": len(stored_vectors) > 0,
    }
    return render(request, "ragAPP/augmentation.html", context)


def similarity_search(query, stored_vectors, top_k=3):
    """Unified similarity search function"""
    if not stored_vectors:
        return []
    
    try:
        from fastembed import TextEmbedding
        model = TextEmbedding(model_name="BAAI/bge-small-en")
        query_vectors = list(model.embed([query]))
        query_embedding = [round(float(x), 6) for x in query_vectors[0]]
        
        # Calculate similarities
        similarities = []
        for vector in stored_vectors:
            query_vec = np.array(query_embedding)
            stored_vec = np.array(vector['embedding'])
            similarity = np.dot(query_vec, stored_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(stored_vec))
            similarities.append((vector, float(similarity)))
        
        # Sort by similarity and get top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]
        
        retrieved_chunks = []
        for i, (vector, similarity) in enumerate(top_results):
            retrieved_chunks.append({
                'rank': i + 1,
                'chunk_id': vector['id'],
                'text': vector['text'],
                'similarity_score': round(similarity, 3),
                'embedding': vector['embedding'][:10]  # Show first 10 dimensions only
            })
        
        return retrieved_chunks
    except Exception as e:
        print(f"Error in similarity search: {str(e)}")
        return []

def generation(request: HttpRequest):
    """Step 7: Generation - LLM Response Generation Demo"""
    # Get stored vectors from session
    stored_vectors = request.session.get('vector_storage', [])
    demo_query = "What is RAG?"
    
    # Use unified similarity search
    retrieved_chunks = similarity_search(demo_query, stored_vectors)
    
    # Create augmented prompt
    augmented_prompt = f"""User Question: {demo_query}

Context:
"""
    
    for i, chunk in enumerate(retrieved_chunks, 1):
        augmented_prompt += f"{i}. {chunk['text']}\n"
    
    augmented_prompt += f"""
Please answer the user's question using the provided context. If the context doesn't contain relevant information, say so clearly."""
    
    # Generate response using the existing generate_response function
    llm_response = generate_response(demo_query, "\n".join([chunk['text'] for chunk in retrieved_chunks]))
    
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
                # Use unified parsing function
                cleaned_text, file_type = parse_document(uploaded_file)
                
                if cleaned_text is None:
                    error = f"❌ {file_type}"  # file_type contains error message
                    print(f"❌ ERROR: {file_type}")
                else:
                    print(f"✅ File processed successfully - {len(cleaned_text)} characters")
                    
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
    model = TextEmbedding(model_name="BAAI/bge-small-en")
    print("✅ FastEmbed model loaded successfully")
    
    # Use unified chunking function
    cleaned_text = clean_text(document.content)
    print(f"📏 Cleaned text length: {len(cleaned_text)} characters")
    
    print("\n✂️ CREATING TEXT CHUNKS")
    chunks = create_chunks(cleaned_text)
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
    """Perform RAG query using database chunks and FastEmbed for similarity search"""
    print(f"\n" + "="*60)
    print(f"🔍 RAG QUERY PROCESSING - Query: '{query}'")
    print("="*60)
    
    # Get all chunks with embeddings from database
    chunks = TextChunk.objects.all()
    print(f"📄 Total chunks in database: {chunks.count()}")
    
    if not chunks.exists():
        print("⚠️ No chunks found in knowledge base")
        return "I don't have any documents in my knowledge base yet. Please upload some documents first.", []
    
    # Convert database chunks to format expected by similarity_search
    stored_vectors = []
    for chunk in chunks:
        stored_vectors.append({
            'id': chunk.id,
            'text': chunk.text,
            'embedding': chunk.embedding
        })
    
    # Use unified similarity search function
    print(f"🔄 Performing similarity search...")
    retrieved_chunks = similarity_search(query, stored_vectors)
    
    # Prepare context for response
    context_text = ""
    db_retrieved_chunks = []
    
    print(f"\n📝 PREPARING CONTEXT FOR RESPONSE")
    for i, chunk_data in enumerate(retrieved_chunks, 1):
        chunk = TextChunk.objects.get(id=chunk_data['chunk_id'])
        context_text += f"{i}. {chunk.text}\n\n"
        db_retrieved_chunks.append({
            'id': chunk.id,
            'text': chunk.text,
            'similarity': chunk_data['similarity_score'],
            'document_title': chunk.document.title
        })
    
    print(f"✅ Context prepared - {len(db_retrieved_chunks)} chunks")
    
    # Generate response using the retrieved context
    print(f"\n🤖 GENERATING RESPONSE")
    response = generate_response(query, context_text)
    print(f"✅ Response generated")
    
    print(f"\n🎉 RAG QUERY COMPLETE")
    print("="*60)
    
    return response, db_retrieved_chunks


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

