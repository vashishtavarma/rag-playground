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
]