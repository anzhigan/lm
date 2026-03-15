from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from app.models.schemas import TextCorpus, SingleText, LSARequest, ProcessingResponse
from app.services.nlp_processor import NLPProcessor
import time
import logging

logger = logging.getLogger(__name__)
router = APIRouter()
nlp_processor = NLPProcessor()


@router.post("/tf-idf", response_model=ProcessingResponse)
async def tfidf_endpoint(corpus: TextCorpus):
    """Вычисление TF-IDF матрицы"""
    start_time = time.time()
    try:
        logger.info(f"Processing TF-IDF for {len(corpus.texts)} texts")
        result = nlp_processor.compute_tfidf(corpus.texts)
        execution_time = time.time() - start_time

        return ProcessingResponse(
            status="success",
            data=result,
            message=f"TF-IDF матрица размером {result['matrix_shape']}",
            execution_time=execution_time
        )
    except Exception as e:
        logger.error(f"Error in TF-IDF: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@router.post("/bag-of-words", response_model=ProcessingResponse)
async def bow_endpoint(corpus: TextCorpus):
    """Вычисление Bag of Words матрицы"""
    start_time = time.time()
    try:
        logger.info(f"Processing Bag of Words for {len(corpus.texts)} texts")
        result = nlp_processor.compute_bow(corpus.texts)
        execution_time = time.time() - start_time

        return ProcessingResponse(
            status="success",
            data=result,
            message=f"BoW матрица размером {result['matrix_shape']}",
            execution_time=execution_time
        )
    except Exception as e:
        logger.error(f"Error in Bag of Words: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@router.post("/lsa", response_model=ProcessingResponse)
async def lsa_endpoint(request: LSARequest):
    """Латентный семантический анализ"""
    start_time = time.time()
    try:
        logger.info(f"Processing LSA for {len(request.texts)} texts with {request.n_components} components")
        result = nlp_processor.compute_lsa(request.texts, request.n_components)
        execution_time = time.time() - start_time

        return ProcessingResponse(
            status="success",
            data=result,
            message=f"LSA выполнен с {len(result['topics'])} темами",
            execution_time=execution_time
        )
    except Exception as e:
        logger.error(f"Error in LSA: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@router.post("/word2vec", response_model=ProcessingResponse)
async def word2vec_endpoint(corpus: TextCorpus):
    """Word2Vec эмбеддинги"""
    start_time = time.time()
    try:
        logger.info(f"Processing Word2Vec for {len(corpus.texts)} texts")
        result = nlp_processor.compute_word2vec(corpus.texts)
        execution_time = time.time() - start_time

        return ProcessingResponse(
            status="success",
            data=result,
            message=f"Созданы эмбеддинги размером {result['embeddings_shape']}",
            execution_time=execution_time
        )
    except Exception as e:
        logger.error(f"Error in Word2Vec: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


# NLTK эндпоинты
@router.get("/nltk/tokenize/{text}")
async def tokenize_text(text: str):
    try:
        logger.info(f"Tokenizing text: {text[:50]}...")
        result = nlp_processor.tokenize(text)
        return result
    except Exception as e:
        logger.error(f"Error in tokenization: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@router.get("/nltk/stem/{text}")
async def stem_text(text: str):
    try:
        logger.info(f"Stemming text: {text[:50]}...")
        result = nlp_processor.stem(text)
        return result
    except Exception as e:
        logger.error(f"Error in stemming: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@router.get("/nltk/lemmatize/{text}")
async def lemmatize_text(text: str):
    try:
        logger.info(f"Lemmatizing text: {text[:50]}...")
        result = nlp_processor.lemmatize(text)
        return result
    except Exception as e:
        logger.error(f"Error in lemmatization: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@router.get("/nltk/pos/{text}")
async def pos_tagging(text: str):
    try:
        logger.info(f"POS tagging text: {text[:50]}...")
        result = nlp_processor.pos_tagging(text)
        return result
    except Exception as e:
        logger.error(f"Error in POS tagging: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@router.get("/nltk/ner/{text}")
async def named_entity_recognition(text: str):
    try:
        logger.info(f"NER for text: {text[:50]}...")
        result = nlp_processor.named_entity_recognition(text)
        return result
    except Exception as e:
        logger.error(f"Error in NER: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@router.post("/nltk/process")
async def full_nltk_processing(text_data: SingleText):
    try:
        logger.info(f"Full NLTK processing for text: {text_data.text[:50]}...")
        result = nlp_processor.tokenize(text_data.text)
        return result
    except Exception as e:
        logger.error(f"Error in full NLTK processing: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )