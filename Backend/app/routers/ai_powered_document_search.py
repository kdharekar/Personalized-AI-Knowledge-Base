from fastapi.responses import JSONResponse
from datetime import date, datetime, timedelta
import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, status, APIRouter
from app.services.index_documents import IndexingService

router = APIRouter()

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_FILE_SIZE = os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024)  # 10 MB default
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}

# Instantiate the service and create the index
indexing_service = IndexingService()


@router.get("/health", status_code=status.HTTP_200_OK)
def health():
    return JSONResponse(content={"status": "ok"}, status_code=status.HTTP_200_OK)

@router.post("/upload/document", status_code=status.HTTP_200_OK)
def upload_document(file: UploadFile = File(...)):
    """
    Endpoint to upload a document. The document is then processed and
    added to a searchable vector index.
    """
    _, ext = os.path.splitext(file.filename)
    ext = ext.lower()

    # Validate file type
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type '{ext}' is not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    file_path = None
    try:
        # Save file to uploads folder
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # --- Index the uploaded document ---

        indexing_service.create_index_from_file(file_path)

        return JSONResponse(
            content={
                "message": "File uploaded and indexed successfully",
                "filename": file.filename,
            },
            status_code=status.HTTP_200_OK
        )

    except Exception as e:
        # If indexing fails, we still have the file, but should report an error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error while processing file: {str(e)}"
        )