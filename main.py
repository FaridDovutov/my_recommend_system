import os
import pickle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

app = FastAPI()

# Настройка CORS, чтобы запросы точно проходили
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Автоматическое определение системных путей внутри сервера Render
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "svd_recommendation_model.pkl")
MOVIES_PATH = os.path.join(BASE_DIR, "u.item")

# 1. Безопасная загрузка модели SVD
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    print(f"ВНИМАНИЕ: Файл модели не найден по пути: {MODEL_PATH}")

# 2. Правильное чтение u.item с учетом кодировки MovieLens
movie_titles = {}
if os.path.exists(MOVIES_PATH):
    # Используем ISO-8859-1, так как в MovieLens есть западноевропейские символы
    with open(MOVIES_PATH, "r", encoding="ISO-8859-1") as f:
        for line in f:
            parts = line.split("|")
            if len(parts) > 1:
                try:
                    movie_id = int(parts[0])
                    movie_title = parts[1]
                    movie_titles[movie_id] = movie_title
                except ValueError:
                    continue
else:
    print(f"ВНИМАНИЕ: Файл u.item не найден по пути: {MOVIES_PATH}")


@app.get("/", response_class=HTMLResponse)
def read_root():
    """Отдает фронтенд"""
    html_file = os.path.join(BASE_DIR, "index.html")
    try:
        with open(html_file, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Хатогӣ: Файли index.html ёфт нашуд!</h1>"


@app.get("/recommend/{user_id}")
def get_recommendations(user_id: int):
    if model is None:
        raise HTTPException(status_code=500, detail="Модели ИС дар сервер боргузорӣ نشده аст.")

    # Если u.item не загрузился, возьмем первые 100 ID для подстраховки, иначе берем из базы
    all_movie_ids = list(movie_titles.keys()) if movie_titles else list(range(1, 100))
    
    recommendations = []
    
    # Расчет предсказаний для всех фильмов
    for m_id in all_movie_ids:
        # В Surprise ID часто обучаются как строки, передаем str(user_id) и str(m_id)
        pred = model.predict(str(user_id), str(m_id))
        recommendations.append({
            "item_id": m_id,
            "estimated_rating": pred.est
        })
    
    # Сортируем по убыванию прогнозной оценки и берем Топ-5
    recommendations.sort(key=lambda x: x["estimated_rating"], reverse=True)
    top_5 = recommendations[:5]
    
    # Добавляем реальные названия фильмов в каждый элемент ТОПа
    for rec in top_5:
        # КЛЮЧЕВОЙ МОМЕНТ: Имя ключа должно быть строго "movie_name"
        rec["movie_name"] = movie_titles.get(rec["item_id"], f"Филми номаълум (ID: {rec['item_id']})")
    
    return {
        "user_id": user_id,
        "recommendations": top_5
    }
