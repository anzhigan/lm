import requests
import json
from typing import List, Dict
import time


class NLPClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()

    def read_texts_from_file(self, filename: str) -> List[str]:
        """Чтение текстов из файла"""
        with open(filename, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def generate_sample_corpus(self) -> List[str]:
        """Генерация примеров текстов для обработки"""
        return [
            "FastAPI это современный веб-фреймворк для Python",
            "Машинное обучение позволяет решать сложные задачи анализа данных",
            "Обработка естественного языка важная область искусственного интеллекта",
            "Python популярный язык для научных вычислений и анализа данных",
            "Нейронные сети показывают отличные результаты в NLP задачах"
        ]

    def test_tfidf(self, texts: List[str]):
        """Тестирование TF-IDF эндпоинта"""
        print("\n" + "=" * 50)
        print("Тестирование TF-IDF")
        print("=" * 50)

        data = {"texts": texts}
        response = self.session.post(f"{self.base_url}/tf-idf", json=data)

        if response.status_code == 200:
            result = response.json()
            print(f"Статус: {result['status']}")
            print(f"Сообщение: {result['message']}")
            print(f"Размер матрицы: {result['data']['matrix_shape']}")
            print(f"Словарь (первые 10 слов): {result['data']['vocabulary'][:10]}")
        else:
            print(f"Ошибка: {response.status_code} - {response.text}")

        return response

    def test_bag_of_words(self, texts: List[str]):
        """Тестирование Bag of Words эндпоинта"""
        print("\n" + "=" * 50)
        print("Тестирование Bag of Words")
        print("=" * 50)

        data = {"texts": texts}
        response = self.session.post(f"{self.base_url}/bag-of-words", json=data)

        if response.status_code == 200:
            result = response.json()
            print(f"Статус: {result['status']}")
            print(f"Сообщение: {result['message']}")
            print(f"Размер словаря: {result['data']['vocabulary_size']}")
        else:
            print(f"Ошибка: {response.status_code} - {response.text}")

    def test_lsa(self, texts: List[str], n_components: int = 3):
        """Тестирование LSA эндпоинта"""
        print("\n" + "=" * 50)
        print("Тестирование LSA")
        print("=" * 50)

        data = {"texts": texts, "n_components": n_components}
        response = self.session.post(f"{self.base_url}/lsa", json=data)

        if response.status_code == 200:
            result = response.json()
            print(f"Статус: {result['status']}")
            print(f"Сообщение: {result['message']}")
            print("Темы:")
            for topic in result['data']['topics']:
                print(f"  Тема {topic['topic_id']}: {topic['top_terms'][:5]}")
        else:
            print(f"Ошибка: {response.status_code} - {response.text}")

    def test_word2vec(self, texts: List[str]):
        """Тестирование Word2Vec эндпоинта"""
        print("\n" + "=" * 50)
        print("Тестирование Word2Vec")
        print("=" * 50)

        data = {"texts": texts}
        response = self.session.post(f"{self.base_url}/word2vec", json=data)

        if response.status_code == 200:
            result = response.json()
            print(f"Статус: {result['status']}")
            print(f"Сообщение: {result['message']}")
            print(f"Размер эмбеддингов: {result['data']['embeddings_shape']}")
        else:
            print(f"Ошибка: {response.status_code} - {response.text}")

    def test_nltk_methods(self, text: str):
        """Тестирование NLTK методов"""
        print("\n" + "=" * 50)
        print("Тестирование NLTK методов")
        print("=" * 50)
        print(f"Исходный текст: {text}")

        # Токенизация
        response = self.session.get(f"{self.base_url}/text_nltk/tokenize/{text}")
        if response.status_code == 200:
            result = response.json()
            print(f"\nТокенизация:")
            print(f"  Слова: {result['word_tokens']}")
            print(f"  Предложения: {result['sentence_tokens']}")

        # Стемминг
        response = self.session.get(f"{self.base_url}/text_nltk/stem/{text}")
        if response.status_code == 200:
            result = response.json()
            print(f"\nСтемминг:")
            print(f"  Стемы: {result['stems']}")

        # Лемматизация
        response = self.session.get(f"{self.base_url}/text_nltk/lemmatize/{text}")
        if response.status_code == 200:
            result = response.json()
            print(f"\nЛемматизация:")
            print(f"  Леммы: {result['lemmas']}")

        # POS tagging
        response = self.session.get(f"{self.base_url}/text_nltk/pos/{text}")
        if response.status_code == 200:
            result = response.json()
            print(f"\nPOS tagging:")
            for word, tag in result['pos_tags'][:10]:
                print(f"  {word}: {tag}")

        # NER
        response = self.session.get(f"{self.base_url}/text_nltk/ner/{text}")
        if response.status_code == 200:
            result = response.json()
            print(f"\nNamed Entities:")
            if result['entities']:
                for entity in result['entities']:
                    print(f"  {entity['entity']}: {entity['label']}")
            else:
                print("  Сущности не найдены")

    def run_all_tests(self, texts: List[str] = None):
        """Запуск всех тестов"""
        if texts is None:
            texts = self.generate_sample_corpus()

        print("Начало тестирования NLP сервиса")
        print(f"Корпус текстов ({len(texts)} документов):")
        for i, text in enumerate(texts, 1):
            print(f"{i}. {text[:50]}...")

        self.test_tfidf(texts)
        self.test_bag_of_words(texts)
        self.test_lsa(texts)
        self.test_word2vec(texts)

        # Тестируем NLTK на первом тексте
        if texts:
            self.test_nltk_methods(texts[0])

        print("\n" + "=" * 50)
        print("Тестирование завершено")
        print("=" * 50)


def main():
    client = NLPClient()

    # Вариант 1: Использовать сгенерированный корпус
    texts = client.generate_sample_corpus()
    client.run_all_tests(texts)

    # Вариант 2: Прочитать из файла (раскомментировать при наличии файла)
    # try:
    #     file_texts = client.read_texts_from_file("sample.txt")
    #     if file_texts:
    #         print("\n" + "="*50)
    #         print("Тестирование с текстами из файла")
    #         print("="*50)
    #         client.run_all_tests(file_texts)
    # except FileNotFoundError:
    #     print("Файл sample.txt не найден, используется сгенерированный корпус")


if __name__ == "__main__":
    # Проверяем доступность сервера
    try:
        response = requests.get("http://localhost:8000")
        if response.status_code == 200:
            print("Сервер доступен, начинаем тестирование...")
            main()
        else:
            print("Сервер вернул ошибку")
    except requests.exceptions.ConnectionError:
        print("Ошибка: Сервер не доступен. Убедитесь, что сервер запущен на http://localhost:8000")