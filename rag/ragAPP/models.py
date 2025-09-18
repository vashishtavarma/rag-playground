from django.db import models
from django.contrib.auth.models import User


class Document(models.Model):
    """Model to store uploaded documents"""
    title = models.CharField(max_length=255)
    file_type = models.CharField(max_length=10, choices=[('txt', 'Text'), ('pdf', 'PDF')])
    content = models.TextField()
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    
    def __str__(self):
        return self.title


# TextChunk model removed - now using ChromaDB for vector storage


class ChatSession(models.Model):
    """Model to store chat sessions"""
    session_id = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_activity = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Chat Session {self.session_id}"


class ChatMessage(models.Model):
    """Model to store chat messages"""
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    message = models.TextField()
    response = models.TextField()
    retrieved_chunks = models.JSONField(default=list)  # Store retrieved chunk IDs
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['timestamp']
    
    def __str__(self):
        return f"Message in {self.session.session_id} at {self.timestamp}"
