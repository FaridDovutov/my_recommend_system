import os
import pickle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

app = FastAPI(title="Системаи тавсиявӣ дар асоси SVD")

# Танзими CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Функция для динамической загрузки реальных названий фильмов
def load_movie_titles():
    movies_dict = {}
    
    # Вариант А: Путь к встроенному датасету Surprise (ml-100k), который скачивается автоматически
    builtin_path = os.path.expanduser('~/.surprise_data/ml-100k/ml-100k/u.item')
    
    # Вариант Б: Если у вас есть локальный файл u.item или movies.csv в папке проекта (укажите ваш путь)
    local_path = 'data/u.item' 
    
    # Определяем, какой файл доступен
    target_path = None
    if os.path.exists(builtin_path):
        target_path = builtin_path
    elif os.path.exists(local_path):
        target_path = local_path

    if target_path:
        try:
            # Файлы MovieLens ml-100k (u.item) разделены символом '|' и имеют кодировку ISO-8859-1
            with open(target_path, 'r', encoding='ISO-8859-1') as f:
                for line in f:
                    fields = line.split('|')
                    if len(fields) > 1:
                        item_id = int(fields[0])
                        movie_name = fields[1]
                        movies_dict[item_id] = movie_name
            print(f"Базаи додаҳо бор карда шуд: {len(movies_dict)} номгӯи филмҳо ёфт шуданд.")
        except Exception as e:
            print(f"Хатогии боркунии файл: {e}")
    else:
        print("Внимание: Файл с названиями фильмов не найден. Будут использованы заглушки ID.")
        
    return movies_dict

# Боркунии ҳамаи номҳои филмҳо аз датасет ҳангоми запуск
movies_dict = load_movie_titles()

# Боркунии модели захирашудаи SVD
MODEL_PATH = 'models/svd_recommendation_model.pkl'
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Модели SVD бо муваффақият бор карда шуд.")
except FileNotFoundError:
    print(f"Хатогӣ: Файли {MODEL_PATH} ёфт нашуд!")
    model = None

@app.get("/", response_class=HTMLResponse)
def read_root():
    """Отдает красивый фронтенд-интерфейс при переходе на главный URL сайта"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Хатогӣ: Файли index.html ёфт нашуд!</h1>"

@app.get("/recommend/{user_id}")
def get_recommendations(user_id: int, top_n: int = 5):
    if model is None:
        raise HTTPException(status_code=500, detail="Модели тавсиявӣ бор карда نشدهаст.")
    
    # Рӯйхати ID-и объектҳо (филмҳо). Барои MovieLens 100k ихтиёрӣ 1682 филмро мегирем
    all_item_ids = list(range(1, 1683))
    
    predictions = []
    for item_id in all_item_ids:
        pred = model.predict(str(user_id), str(item_id))
        predictions.append((item_id, pred.est))
    
    # Сортировка аз рӯи баҳогузории баландтарин
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_predictions = predictions[:top_n]
    
    recommendations = []
    for item_id, estimated_rating in top_predictions:
        # Извлекаем реальное название из словаря датасета. Если его там нет — сработает локализованная заглушка
        movie_name = movies_dict.get(item_id, f"Филми дорои ID {item_id}")
        recommendations.append({
            "item_id": item_id,
            "movie_name": movie_name,
            "estimated_rating": round(estimated_rating, 2)
        })
        
    return {
        "user_id": user_id,
        "recommendations": recommendations
    }