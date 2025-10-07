from fastapi.responses import JSONResponse, HTMLResponse
from datetime import date, datetime, timedelta
import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, status, APIRouter, Request
from app.services.index_documents import IndexingService
from app.services.search_documents import SearchService
from app.models.request_models import SearchQuery, FeedbackRequest
from app.services.feedback_logger import FeedbackLogger
import uuid
from fastapi.templating import Jinja2Templates


# Point to the folder where your HTML templates are stored
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "../../templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


feedback_logger = FeedbackLogger()
router = APIRouter()

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_FILE_SIZE = os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024)  # 10 MB default
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}

# Instantiate the service and create the index
indexing_service = IndexingService()
search_service = SearchService()

@router.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


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

@router.post("/search/doc")
async def search_document(query: SearchQuery):
    """
    Endpoint to search the indexed documents.
    Accepts a JSON body with a "query" field.
    """
    try:
        # Use the pre-initialized search_service instance
        search_id = str(uuid.uuid4())
        # 2. Get the result from the search service
        result = search_service.search(query.query)
        # 3. Add the search_id to the response
        result['search_id'] = search_id
        return JSONResponse(content=result)
    except Exception as e:
        # This is a fallback for unexpected errors in the service
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during the search: {str(e)}"
        )

@router.post("/feedback", status_code=status.HTTP_201_CREATED)
async def submit_feedback(feedback: FeedbackRequest):
    """
    Endpoint for users to submit feedback on an answer.
    The feedback is logged for future analysis and pipeline improvement.
    """
    try:
        # The Pydantic model has already validated the input
        # Log the feedback using our service
        feedback_logger.log(feedback.model_dump())
        
        return {"message": "Feedback received successfully. Thank you!"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to log feedback: {str(e)}"
        )