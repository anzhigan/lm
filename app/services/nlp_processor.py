import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from app.services.text_preprocessor import TextPreprocessor
import logging

logger = logging.getLogger(__name__)


class NLPProcessor:
    def __init__(self):
        # Загрузка ресурсов NLTK
        self._download_nltk_resources()

        # Инициализация компонентов
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.preprocessor = TextPreprocessor()

    def _download_nltk_resources(self):
        """Загрузка необходимых ресурсов NLTK"""
        resources = [
            'punkt', 'averaged_perceptron_tagger',
            'maxent_ne_chunker', 'words', 'wordnet', 'stopwords'
        ]
        for resource in resources:
            try:
                nltk.download(resource, quiet=True)
            except Exception as e:
                logger.warning(f"Failed to download {resource}: {e}")

    def compute_tfidf(self, texts):
        """Вычисление TF-IDF матрицы"""
        processed_texts = [self.preprocessor.preprocess_text(t) for t in texts]

        # Создаем словарь уникальных слов
        all_words = set()
        for text in processed_texts:
            words = text.split()
            all_words.update(words)

        word_to_idx = {word: i for i, word in enumerate(sorted(all_words))}
        n_docs = len(processed_texts)
        n_terms = len(word_to_idx)

        # Создаем матрицу частот терминов
        tf_matrix = np.zeros((n_docs, n_terms))
        for i, text in enumerate(processed_texts):
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

        return {
            "vocabulary": list(word_to_idx.keys()),
            "matrix_shape": tfidf_matrix.shape,
            "matrix": tfidf_matrix.tolist()
        }

    def compute_bow(self, texts):
        """Вычисление Bag of Words матрицы"""
        processed_texts = [self.preprocessor.preprocess_text(t) for t in texts]

        all_words = set()
        for text in processed_texts:
            words = text.split()
            all_words.update(words)

        word_to_idx = {word: i for i, word in enumerate(sorted(all_words))}
        n_docs = len(processed_texts)
        n_terms = len(word_to_idx)

        bow_matrix = np.zeros((n_docs, n_terms))
        for i, text in enumerate(processed_texts):
            words = text.split()
            for word in words:
                if word in word_to_idx:
                    bow_matrix[i, word_to_idx[word]] += 1

        return {
            "vocabulary": list(word_to_idx.keys()),
            "vocabulary_size": n_terms,
            "matrix_shape": bow_matrix.shape,
            "matrix": bow_matrix.tolist()
        }

    def compute_lsa(self, texts, n_components=5):
        """Вычисление LSA"""
        processed_texts = [self.preprocessor.preprocess_text(t) for t in texts]

        vectorizer = TfidfVectorizer(max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(processed_texts)

        n_components = min(n_components, tfidf_matrix.shape[1])
        lsa = TruncatedSVD(n_components=n_components, random_state=42)
        lsa_matrix = lsa.fit_transform(tfidf_matrix)

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

        return {
            "topics": topics,
            "transformed_matrix_shape": lsa_matrix.shape,
            "transformed_matrix": lsa_matrix.tolist()
        }

    def compute_word2vec(self, texts, vector_size=100):
        """Упрощенная версия Word2Vec"""
        processed_texts = [self.preprocessor.preprocess_text(t) for t in texts]

        vectorizer = TfidfVectorizer(max_features=vector_size)
        embeddings = vectorizer.fit_transform(processed_texts).toarray()

        return {
            "vocabulary": list(vectorizer.get_feature_names_out()),
            "embeddings_shape": embeddings.shape,
            "embeddings": embeddings.tolist()
        }

    # NLTK методы
    def tokenize(self, text):
        words = word_tokenize(text, language='russian')
        sentences = sent_tokenize(text, language='russian')
        return {
            "original_text": text,
            "word_tokens": words,
            "sentence_tokens": sentences,
            "word_count": len(words),
            "sentence_count": len(sentences)
        }

    def stem(self, text):
        words = word_tokenize(text)
        stems = [self.stemmer.stem(word) for word in words]
        return {
            "original_text": text,
            "original_words": words,
            "stems": stems,
            "stemmed_text": " ".join(stems)
        }

    def lemmatize(self, text):
        words = word_tokenize(text)
        lemmas = [self.lemmatizer.lemmatize(word) for word in words]
        return {
            "original_text": text,
            "original_words": words,
            "lemmas": lemmas,
            "lemmatized_text": " ".join(lemmas)
        }

    def pos_tagging(self, text):
        words = word_tokenize(text)
        pos_tags = pos_tag(words)
        return {
            "original_text": text,
            "pos_tags": pos_tags
        }

    def named_entity_recognition(self, text):
        words = word_tokenize(text)
        pos_tags = pos_tag(words)
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
            "entities": entities
        }