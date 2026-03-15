# NLP Processing Service

Сервис для обработки естественного языка с веб-интерфейсом.

## Возможности

- TF-IDF векторизация
- Bag of Words
- Латентный семантический анализ (LSA)
- Word2Vec эмбеддинги
- Токенизация
- Стемминг и лемматизация
- POS tagging
- Named Entity Recognition (NER)

## Установка

```bash
1. Клонировать репозиторий:
git clone ..
cd lm

2. Создать виртуальное окружение:
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

3. Установить зависимости:
pip install -r requirements.txt

4. Запустить сервер:
python -m app.main

5. Открыть браузер и перейти по адресу:
http://localhost:8000

6. Для запуска клиента (альтернативный способ):
python client/client.py