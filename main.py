from fastapi import FastAPI

# Создаем экземпляр приложения FastAPI
app = FastAPI()

# Определяем "маршрут" (endpoint) для GET запросов по адресу "/"
@app.get("/")
def read_root():
    return {"message": "Hello World"}

# Еще один endpoint для получения информации о пользователе по его ID
@app.get("/users/{user_id}")
def read_user(user_id: int, query_param: str = None):
    return {"user_id": user_id, "query_param": query_param}