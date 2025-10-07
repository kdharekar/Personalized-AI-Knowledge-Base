import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

PERSIST_DIRECTORY = "./vectordb" 
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class IndexingService:
    """
    A service to handle the creation and updating of a vector index
    for searchable documents.
    """
    def __init__(self):
        """
        Initializes the IndexingService with an embedding model and a
        persistent vector store.
        """
        #self.embedding_function = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Initialize the Chroma vector store with a persistence directory
        self.vector_store = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=self.embedding_function
        )
        print(f"Vector store initialized. Total documents in store: {self.vector_store._collection.count()}")


    def _get_loader_for_file(self, file_path: str, file_ext: str):
        """
        Returns the appropriate LangChain document loader based on the file extension.
        """
        if file_ext == ".pdf":
            return PyPDFLoader(file_path)
        elif file_ext == ".txt":
            return TextLoader(file_path)
        elif file_ext == ".md":
            return UnstructuredMarkdownLoader(file_path)
        else:
            print(f"Warning: No specific loader for '{file_ext}'. Using generic TextLoader.")
            return TextLoader(file_path)

    def create_index_from_file(self, file_path: str):
        """
        Processes a single file, splits it into chunks, creates embeddings,
        and adds them to the vector store.

        Args:
            file_path (str): The path to the file to be indexed.
        """
        print(f"Starting to index file: {file_path}")
        try:
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()

            loader = self._get_loader_for_file(file_path, ext)
            documents = loader.load()
            
            if not documents:
                print(f"Warning: No content loaded from {file_path}. Skipping.")
                return

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            chunks = text_splitter.split_documents(documents)
            
            if not chunks:
                print(f"Warning: Could not split document into chunks: {file_path}. Skipping.")
                return

            print(f"Loaded {len(documents)} document(s) and split into {len(chunks)} chunks.")
            self.vector_store.add_documents(chunks)

            print(f"Successfully indexed {len(chunks)} chunks from {os.path.basename(file_path)}.")
            print(f"Vector store now contains {self.vector_store._collection.count()} total documents.")

        except Exception as e:
            print(f"Error indexing file {file_path}: {e}")
            raise