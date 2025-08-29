from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("input-parsing/", views.input_parsing, name="input_parsing"),
    path("chunking/", views.chunking, name="chunking"),
    path("vector-embedding/", views.vector_embedding, name="vector_embedding"),
    path("vector-storage/", views.vector_storage, name="vector_storage"),
    path("retrieval/", views.retrieval, name="retrieval"),
    path("augmentation/", views.augmentation, name="augmentation"),
    path("generation/", views.generation, name="generation"),
    path("document-text-interface/", views.document_text_interface, name="document_text_interface"),
    path("knowledge-base/", views.knowledge_base, name="knowledge_base"),
    path("chat-interface/", views.chat_interface, name="chat_interface"),
    path("chat-query/", views.chat_query, name="chat_query"),
]