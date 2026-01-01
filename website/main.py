from fastapi import FastAPI, Request, Form, HTTPException, status, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env")

app = FastAPI(title="Jivascope")

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY", "fallback-secret-key")
)

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

VALID_USERNAME = os.getenv("APP_USERNAME", "admin")
VALID_PASSWORD = os.getenv("APP_PASSWORD", "password123")


@app.on_event("startup")
async def startup_event():
    """Load model on server startup."""
    from predictor import predictor
    print("=" * 50)
    print("JIVASCOPE SERVER STARTING")
    print("=" * 50)
    predictor.load_model()
    print("=" * 50)
    print("SERVER READY TO ACCEPT REQUESTS")
    print("=" * 50)


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    if request.session.get("authenticated"):
        return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = None):
    if request.session.get("authenticated"):
        return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse("login.html", {"request": request, "error": error})


@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if username == VALID_USERNAME and password == VALID_PASSWORD:
        request.session["authenticated"] = True
        request.session["username"] = username
        return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse(
        "login.html", 
        {"request": request, "error": "Invalid username or password"},
        status_code=status.HTTP_401_UNAUTHORIZED
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    if not request.session.get("authenticated"):
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    username = request.session.get("username", "User")
    return templates.TemplateResponse("dashboard.html", {"request": request, "username": username})


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)


@app.post("/api/analyze")
async def analyze_audio(request: Request, audio: UploadFile = File(...)):
    """Analyze uploaded audio file for heart sound and murmur detection."""
    if not request.session.get("authenticated"):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    valid_extensions = {'.wav', '.mp3', '.ogg', '.flac'}
    file_ext = Path(audio.filename).suffix.lower() if audio.filename else ''
    
    if file_ext not in valid_extensions:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": f"Invalid file type. Supported formats: {', '.join(valid_extensions)}"
            }
        )
    
    try:
        from predictor import predictor
        audio_bytes = await audio.read()
        result = predictor.predict(audio_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

