from django.contrib import admin
from .models import Document, ChatSession, ChatMessage


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ['title', 'file_type', 'processed', 'uploaded_at']
    list_filter = ['file_type', 'processed', 'uploaded_at']
    search_fields = ['title', 'content']
    readonly_fields = ['uploaded_at']


# TextChunk admin removed - now using ChromaDB for vector storage


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ['session_id', 'created_at', 'message_count']
    search_fields = ['session_id']
    readonly_fields = ['created_at']
    
    def message_count(self, obj):
        return obj.messages.count()
    message_count.short_description = 'Messages'


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ['id', 'session', 'message_preview', 'response_preview', 'timestamp', 'chunk_count']
    list_filter = ['timestamp', 'session']
    search_fields = ['message', 'response', 'session__session_id']
    readonly_fields = ['timestamp', 'retrieved_chunks']
    
    def message_preview(self, obj):
        return obj.message[:50] + "..." if len(obj.message) > 50 else obj.message
    message_preview.short_description = 'Message'
    
    def response_preview(self, obj):
        return obj.response[:50] + "..." if len(obj.response) > 50 else obj.response
    response_preview.short_description = 'Response'
    
    def chunk_count(self, obj):
        return len(obj.retrieved_chunks) if obj.retrieved_chunks else 0
    chunk_count.short_description = 'Retrieved Chunks'
