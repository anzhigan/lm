from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.api.endpoints import router as api_router
import logging
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управление жизненным циклом приложения
    """
    # Startup
    logger.info("NLP Service starting up...")
    yield
    # Shutdown
    logger.info("NLP Service shutting down...")

# Создаем приложение с lifespan
app = FastAPI(
    title="NLP Processing Service",
    description="Сервис для обработки естественного языка",
    version="1.0.0",
    lifespan=lifespan
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене заменить на конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем статические файлы
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Подключаем API роуты
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def serve_frontend():
    """Отдаем главную страницу"""
    return FileResponse("app/static/index.html")

@app.get("/api/v1/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {"status": "healthy", "service": "NLP Processing Service"}

# Этот блок нужен только для прямого запуска файла
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )