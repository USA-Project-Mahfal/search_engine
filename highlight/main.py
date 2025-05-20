# app/main.py
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .services.document_service import DocumentService

# Define the base path to your document directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNK_INPUT_DIR = os.path.join(BASE_DIR, "chunk_input")

# Create FastAPI app
app = FastAPI(title="Document Viewer API")

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Set up templates
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Initialize document service
document_service = DocumentService(CHUNK_INPUT_DIR)

class DocumentRequest(BaseModel):
    category: str
    doc_name: str
    text: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/document")
async def get_document(document_request: DocumentRequest):
    """
    Get a document with highlighted text
    
    Args:
        document_request (DocumentRequest): Document request with category, name and text to highlight
    
    Returns:
        dict: Document with highlighted text
    """
    result = document_service.get_highlighted_document(
        document_request.category,
        document_request.doc_name,
        document_request.text
    )
    
    if result.get("status") == "error":
        raise HTTPException(status_code=404, detail=result["message"])
        
    return result

@app.get("/api/categories")
async def get_categories():
    """
    Get all available document categories
    
    Returns:
        list: List of document categories
    """
    try:
        categories = [d for d in os.listdir(CHUNK_INPUT_DIR) 
                     if os.path.isdir(os.path.join(CHUNK_INPUT_DIR, d))]
        return {"categories": categories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents/{category}")
async def get_documents(category: str):
    """
    Get all documents in a category
    
    Args:
        category (str): Document category
    
    Returns:
        list: List of documents in the category
    """
    try:
        category_path = os.path.join(CHUNK_INPUT_DIR, category)
        if not os.path.isdir(category_path):
            raise HTTPException(status_code=404, detail=f"Category not found: {category}")
            
        documents = [doc for doc in os.listdir(category_path) 
                    if os.path.isfile(os.path.join(category_path, doc))]
        return {"documents": documents}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))