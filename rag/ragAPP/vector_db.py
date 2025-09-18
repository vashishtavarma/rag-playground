"""
ChromaDB Vector Database Service for RAG Application
Handles vector storage, retrieval, and management using ChromaDB
"""

import chromadb
from typing import List, Dict, Any
import uuid
import os
from fastembed import TextEmbedding


class ChromaVectorDB:
    """ChromaDB wrapper for vector storage and retrieval"""
    
    def __init__(self, persist_directory: str = "chroma_db"):
        """Initialize ChromaDB client with persistent storage"""
        self.persist_directory = persist_directory
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="rag_documents",
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        # Initialize embedding model
        self.embedding_model = TextEmbedding(model_name="BAAI/bge-small-en")
        
        print(f"✅ ChromaDB initialized with collection: {self.collection.name}")
        print(f"📁 Persistent storage: {persist_directory}")
    
    def add_documents(self, texts: List[str], metadatas: List[Dict] = None, ids: List[str] = None) -> List[str]:
        """Add documents to the vector database"""
        if not texts:
            return []
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        
        # Generate embeddings
        print(f"🧠 Generating embeddings for {len(texts)} documents...")
        embeddings = list(self.embedding_model.embed(texts))
        embeddings_list = [[float(x) for x in embedding] for embedding in embeddings]
        
        # Prepare metadata
        if metadatas is None:
            metadatas = [{"text_length": len(text)} for text in texts]
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings_list,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Added {len(texts)} documents to ChromaDB")
        return ids
    
    def query_similar(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """Query for similar documents"""
        # Generate query embedding
        query_embeddings = list(self.embedding_model.embed([query_text]))
        query_embedding = [float(x) for x in query_embeddings[0]]
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, (doc, metadata, distance, doc_id) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0],
                results['ids'][0]
            )):
                formatted_results.append({
                    'id': doc_id,
                    'text': doc,
                    'metadata': metadata,
                    'similarity_score': 1 - distance,  # Convert distance to similarity
                    'rank': i + 1
                })
        
        return {
            'query': query_text,
            'results': formatted_results,
            'total_results': len(formatted_results)
        }
    
    def get_all_documents(self) -> Dict[str, Any]:
        """Get all documents in the collection organized by document title"""
        try:
            results = self.collection.get()
            
            # Organize documents by document title
            documents_by_title = {}
            all_documents = []
            
            if results['documents']:
                for i, (doc_id, doc, metadata) in enumerate(zip(
                    results['ids'],
                    results['documents'],
                    results['metadatas']
                )):
                    doc_title = metadata.get('document_title', 'Unknown Document')
                    chunk_index = metadata.get('chunk_index', 0)
                    
                    document_info = {
                        'id': doc_id,
                        'text': doc,
                        'metadata': metadata,
                        'document_title': doc_title,
                        'chunk_index': chunk_index,
                        'text_length': len(doc)
                    }
                    
                    # Group by document title
                    if doc_title not in documents_by_title:
                        documents_by_title[doc_title] = []
                    documents_by_title[doc_title].append(document_info)
                    
                    all_documents.append(document_info)
            
            # Sort chunks within each document by chunk_index
            for title in documents_by_title:
                documents_by_title[title].sort(key=lambda x: x['chunk_index'])
            
            return {
                'documents': all_documents,
                'documents_by_title': documents_by_title,
                'total_count': len(all_documents),
                'total_documents': len(documents_by_title),
                'collection_name': self.collection.name
            }
        except Exception as e:
            print(f"Error getting all documents: {str(e)}")
            return {
                'documents': [], 
                'documents_by_title': {},
                'total_count': 0, 
                'total_documents': 0,
                'collection_name': self.collection.name
            }
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID"""
        try:
            self.collection.delete(ids=[doc_id])
            print(f"✅ Deleted document: {doc_id}")
            return True
        except Exception as e:
            print(f"❌ Error deleting document {doc_id}: {str(e)}")
            return False
    
    def clear_all(self) -> bool:
        """Clear all documents from the collection"""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name="rag_documents")
            self.collection = self.client.get_or_create_collection(
                name="rag_documents",
                metadata={"hnsw:space": "cosine"}
            )
            print("✅ Cleared all documents from ChromaDB")
            return True
        except Exception as e:
            print(f"❌ Error clearing collection: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            return {
                'total_documents': count,
                'collection_name': self.collection.name,
                'embedding_model': "BAAI/bge-small-en",
                'embedding_dimensions': 384,
                'similarity_metric': "cosine"
            }
        except Exception as e:
            print(f"Error getting collection stats: {str(e)}")
            return {
                'total_documents': 0,
                'collection_name': self.collection.name,
                'embedding_model': "BAAI/bge-small-en",
                'embedding_dimensions': 384,
                'similarity_metric': "cosine"
            }
    
    def add_document_chunks(self, document_title: str, chunks: List[str]) -> List[str]:
        """Add document chunks with metadata"""
        if not chunks:
            return []
        
        # Prepare metadata for each chunk
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document_title}_{i}_{str(uuid.uuid4())[:8]}"
            ids.append(chunk_id)
            metadatas.append({
                'document_title': document_title,
                'chunk_index': i,
                'text_length': len(chunk),
                'chunk_type': 'document_chunk'
            })
        
        return self.add_documents(chunks, metadatas, ids)


# Global instance
vector_db = None

def get_vector_db() -> ChromaVectorDB:
    """Get or create global vector database instance"""
    global vector_db
    if vector_db is None:
        # Use project-specific directory
        persist_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'chroma_storage')
        vector_db = ChromaVectorDB(persist_directory=persist_dir)
    return vector_db
