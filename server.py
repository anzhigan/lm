from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk
import re
import uvicorn
import logging

# Загрузка необходимых ресурсов NLTK
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    print("NLTK resources already downloaded or failed to download")

app = FastAPI(title="NLP Processing Service")


# Модели данных
class TextCorpus(BaseModel):
    texts: List[str]
    document_ids: List[str] = None


class ProcessingResponse(BaseModel):
    status: str
    data: Any
    message: str = None


# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация NLP компонентов
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


# Утилиты для предобработки
def preprocess_text(text: str) -> str:
    """Базовая предобработка текста"""
    # Приводим к нижнему регистру
    text = text.lower()
    # Удаляем специальные символы
    text = re.sub(r'[^\w\s]', '', text)
    # Удаляем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@app.get("/", response_class=HTMLResponse)
async def root():
    """Главная страница с информацией о сервисе"""
    html_content = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>NLP Processing Service</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #2c3e50; }
                .endpoint { background-color: #ecf0f1; padding: 10px; margin: 10px 0; border-radius: 5px; }
                code { background-color: #bdc3c7; padding: 2px 5px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <h1>NLP Processing Service</h1>
            <p>Доступные эндпоинты:</p>
            <div class="endpoint">
                <strong>POST /tf-idf</strong> - Вычисление TF-IDF матрицы
            </div>
            <div class="endpoint">
                <strong>POST /bag-of-words</strong> - Создание мешка слов
            </div>
            <div class="endpoint">
                <strong>POST /lsa</strong> - Латентный семантический анализ
            </div>
            <div class="endpoint">
                <strong>POST /word2vec</strong> - Word2Vec эмбеддинги
            </div>
            <div class="endpoint">
                <strong>GET /text_nltk/tokenize/{text}</strong> - Токенизация
            </div>
            <div class="endpoint">
                <strong>GET /text_nltk/stem/{text}</strong> - Стемминг
            </div>
            <div class="endpoint">
                <strong>GET /text_nltk/lemmatize/{text}</strong> - Лемматизация
            </div>
            <div class="endpoint">
                <strong>GET /text_nltk/pos/{text}</strong> - POS tagging
            </div>
            <div class="endpoint">
                <strong>GET /text_nltk/ner/{text}</strong> - Named Entity Recognition
            </div>
            <div class="endpoint">
                <strong>POST /text_nltk/process</strong> - Полная обработка текста
            </div>
        </body>
    </html>
    """
    return html_content


# TF-IDF эндпоинт
@app.post("/tf-idf", response_model=ProcessingResponse)
async def calculate_tf_idf(corpus: TextCorpus):
    """
    Вычисление TF-IDF матрицы с использованием numpy
    """
    try:
        texts = [preprocess_text(text) for text in corpus.texts]

        # Создаем словарь уникальных слов
        all_words = set()
        for text in texts:
            words = text.split()
            all_words.update(words)

        word_to_idx = {word: i for i, word in enumerate(sorted(all_words))}
        n_docs = len(texts)
        n_terms = len(word_to_idx)

        # Создаем матрицу частот терминов
        tf_matrix = np.zeros((n_docs, n_terms))
        for i, text in enumerate(texts):
            words = text.split()
            for word in words:
                if word in word_to_idx:
                    tf_matrix[i, word_to_idx[word]] += 1

        # Вычисляем IDF
        df = np.sum(tf_matrix > 0, axis=0)
        idf = np.log((n_docs + 1) / (df + 1)) + 1

        # Вычисляем TF-IDF
        tfidf_matrix = tf_matrix * idf

        # Нормализация
        norms = np.sqrt(np.sum(tfidf_matrix ** 2, axis=1, keepdims=True))
        tfidf_matrix = tfidf_matrix / (norms + 1e-10)

        # Формируем результат
        result = {
            "vocabulary": list(word_to_idx.keys()),
            "matrix_shape": tfidf_matrix.shape,
            "matrix": tfidf_matrix.tolist()
        }

        return ProcessingResponse(
            status="success",
            data=result,
            message=f"TF-IDF матрица размером {tfidf_matrix.shape}"
        )
    except Exception as e:
        logger.error(f"Error in TF-IDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Bag of Words эндпоинт
@app.post("/bag-of-words", response_model=ProcessingResponse)
async def bag_of_words(corpus: TextCorpus):
    """
    Создание мешка слов с использованием numpy
    """
    try:
        texts = [preprocess_text(text) for text in corpus.texts]

        # Создаем словарь
        all_words = set()
        for text in texts:
            words = text.split()
            all_words.update(words)

        word_to_idx = {word: i for i, word in enumerate(sorted(all_words))}
        n_docs = len(texts)
        n_terms = len(word_to_idx)

        # Создаем матрицу мешка слов
        bow_matrix = np.zeros((n_docs, n_terms))
        for i, text in enumerate(texts):
            words = text.split()
            for word in words:
                if word in word_to_idx:
                    bow_matrix[i, word_to_idx[word]] += 1

        result = {
            "vocabulary": list(word_to_idx.keys()),
            "vocabulary_size": n_terms,
            "matrix_shape": bow_matrix.shape,
            "matrix": bow_matrix.tolist()
        }

        return ProcessingResponse(
            status="success",
            data=result,
            message=f"BoW матрица размером {bow_matrix.shape} с {n_terms} уникальными словами"
        )
    except Exception as e:
        logger.error(f"Error in Bag of Words: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# LSA эндпоинт
@app.post("/lsa", response_model=ProcessingResponse)
async def latent_semantic_analysis(corpus: TextCorpus, n_components: int = 5):
    """
    Латентный семантический анализ с использованием sklearn
    """
    try:
        texts = [preprocess_text(text) for text in corpus.texts]

        # Создаем TF-IDF матрицу
        vectorizer = TfidfVectorizer(max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(texts)

        # Применяем LSA
        lsa = TruncatedSVD(n_components=min(n_components, tfidf_matrix.shape[1]), random_state=42)
        lsa_matrix = lsa.fit_transform(tfidf_matrix)

        # Получаем темы
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for i, comp in enumerate(lsa.components_):
            top_terms_idx = comp.argsort()[-10:][::-1]
            top_terms = [feature_names[idx] for idx in top_terms_idx]
            topics.append({
                "topic_id": i,
                "explained_variance": float(lsa.explained_variance_ratio_[i]),
                "top_terms": top_terms
            })

        result = {
            "topics": topics,
            "transformed_matrix_shape": lsa_matrix.shape,
            "transformed_matrix": lsa_matrix.tolist()
        }

        return ProcessingResponse(
            status="success",
            data=result,
            message=f"LSA выполнен с {len(topics)} темами"
        )
    except Exception as e:
        logger.error(f"Error in LSA: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Word2Vec эндпоинт (упрощенная версия)
@app.post("/word2vec", response_model=ProcessingResponse)
async def word2vec_embeddings(corpus: TextCorpus, vector_size: int = 100):
    """
    Word2Vec эмбеддинги (упрощенная версия на основе sklearn)
    """
    try:
        texts = [preprocess_text(text) for text in corpus.texts]

        # Используем TfidfVectorizer для создания эмбеддингов документов
        vectorizer = TfidfVectorizer(max_features=vector_size)
        embeddings = vectorizer.fit_transform(texts).toarray()

        result = {
            "vocabulary": list(vectorizer.get_feature_names_out()),
            "embeddings_shape": embeddings.shape,
            "embeddings": embeddings.tolist()
        }

        return ProcessingResponse(
            status="success",
            data=result,
            message=f"Созданы эмбеддинги размером {embeddings.shape}"
        )
    except Exception as e:
        logger.error(f"Error in Word2Vec: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# NLTK эндпоинты
@app.get("/text_nltk/tokenize/{text}")
async def tokenize_text(text: str):
    """Токенизация текста"""
    try:
        words = word_tokenize(text, language='russian')
        sentences = sent_tokenize(text, language='russian')
        return {
            "original_text": text,
            "word_tokens": words,
            "sentence_tokens": sentences,
            "word_count": len(words),
            "sentence_count": len(sentences)
        }
    except Exception as e:
        # Fallback to English tokenizer
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        return {
            "original_text": text,
            "word_tokens": words,
            "sentence_tokens": sentences,
            "word_count": len(words),
            "sentence_count": len(sentences)
        }


@app.get("/text_nltk/stem/{text}")
async def stem_text(text: str):
    """Стемминг текста"""
    words = word_tokenize(text)
    stems = [stemmer.stem(word) for word in words]
    return {
        "original_text": text,
        "original_words": words,
        "stems": stems,
        "stemmed_text": " ".join(stems)
    }


@app.get("/text_nltk/lemmatize/{text}")
async def lemmatize_text(text: str):
    """Лемматизация текста"""
    words = word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(word) for word in words]
    return {
        "original_text": text,
        "original_words": words,
        "lemmas": lemmas,
        "lemmatized_text": " ".join(lemmas)
    }


@app.get("/text_nltk/pos/{text}")
async def pos_tagging(text: str):
    """Part-of-Speech tagging"""
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    return {
        "original_text": text,
        "pos_tags": pos_tags
    }


@app.get("/text_nltk/ner/{text}")
async def named_entity_recognition(text: str):
    """Named Entity Recognition"""
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    ner_tree = ne_chunk(pos_tags)

    # Извлекаем именованные сущности
    entities = []
    for chunk in ner_tree:
        if hasattr(chunk, 'label'):
            entities.append({
                "entity": " ".join([token for token, pos in chunk.leaves()]),
                "label": chunk.label()
            })

    return {
        "original_text": text,
        "entities": entities
    }


@app.post("/text_nltk/process")
async def full_nltk_processing(text_data: Dict[str, str]):
    """Полная NLTK обработка текста"""
    text = text_data.get("text", "")

    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    stems = [stemmer.stem(word) for word in words]
    lemmas = [lemmatizer.lemmatize(word) for word in words]
    pos_tags = pos_tag(words)

    # NER
    ner_tree = ne_chunk(pos_tags)
    entities = []
    for chunk in ner_tree:
        if hasattr(chunk, 'label'):
            entities.append({
                "entity": " ".join([token for token, pos in chunk.leaves()]),
                "label": chunk.label()
            })

    return {
        "original_text": text,
        "word_count": len(words),
        "sentence_count": len(sentences),
        "sentences": sentences,
        "words": words,
        "stems": stems,
        "lemmas": lemmas,
        "pos_tags": pos_tags,
        "entities": entities
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)