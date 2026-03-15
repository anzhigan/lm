import re
from typing import List


class TextPreprocessor:
    @staticmethod
    def preprocess_text(text: str) -> str:
        """Базовая предобработка текста"""
        # Приводим к нижнему регистру
        text = text.lower()
        # Удаляем специальные символы
        text = re.sub(r'[^\w\s]', '', text)
        # Удаляем лишние пробелы
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Разбиение на предложения"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]