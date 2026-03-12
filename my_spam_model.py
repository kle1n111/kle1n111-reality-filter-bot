# my_spam_model.py (исправленная версия)
import pandas as pd
import numpy as np
import sqlite3
import joblib
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
import string
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Скачиваем стоп-слова
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


class MySpamModel:
    def __init__(self, model_path='my_model.pkl', db_path='training_data.db'):
        self.model_path = model_path
        self.db_path = db_path
        self.model = None
        self.categories = ['spam', 'urgent', 'work', 'family', 'personal', 'other']

        # Стоп-слова
        try:
            self.stop_words_ru = set(stopwords.words('russian'))
            self.stop_words_en = set(stopwords.words('english'))
        except:
            self.stop_words_ru = set()
            self.stop_words_en = set()

        # Создаем таблицу при инициализации
        self._init_database()

        # Пытаемся загрузить существующую модель
        self.load_model()

    def _init_database(self):
        """Создает таблицу в базе данных, если её нет"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS training_samples
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          text TEXT NOT NULL,
                          category TEXT NOT NULL,
                          marked_by INTEGER,
                          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
            conn.commit()
            conn.close()
            logger.info("✅ База данных инициализирована")
        except Exception as e:
            logger.error(f"❌ Ошибка при инициализации БД: {e}")

    def preprocess_text(self, text):
        """Очистка текста"""
        if not isinstance(text, str):
            text = str(text)

        # Приводим к нижнему регистру
        text = text.lower()

        # Удаляем пунктуацию и цифры
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)

        # Удаляем лишние пробелы
        text = ' '.join(text.split())

        # Удаляем стоп-слова (если доступны)
        if self.stop_words_ru or self.stop_words_en:
            words = text.split()
            words = [w for w in words if w not in self.stop_words_ru and w not in self.stop_words_en]
            text = ' '.join(words)

        return text

    def load_training_data_from_db(self):
        """Загружает данные из БД с обработкой ошибок"""
        try:
            # Проверяем, существует ли файл БД
            if not os.path.exists(self.db_path):
                logger.warning("Файл базы данных не найден. Использую примеры для начала.")
                return self._create_sample_data()

            conn = sqlite3.connect(self.db_path)

            # Проверяем, существует ли таблица
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='training_samples'")
            if not cursor.fetchone():
                logger.warning("Таблица training_samples не найдена. Использую примеры для начала.")
                conn.close()
                return self._create_sample_data()

            # Загружаем данные
            df = pd.read_sql_query("""
                SELECT text, category, created_at 
                FROM training_samples 
                ORDER BY created_at DESC
            """, conn)

            conn.close()

            if len(df) == 0:
                logger.warning("Нет размеченных данных. Использую примеры для начала.")
                return self._create_sample_data()

            logger.info(f"✅ Загружено {len(df)} размеченных сообщений из БД")
            return df

        except Exception as e:
            logger.error(f"❌ Ошибка при загрузке данных из БД: {e}")
            return self._create_sample_data()

    def _create_sample_data(self):
        """Создает примеры для первого обучения"""
        logger.info("Создаю примеры данных для первого обучения...")

        samples = {
            'text': [
                # Спам
                "Купите срочно акции, огромная прибыль!",
                "Вы выиграли миллион! Перейдите по ссылке",
                "Заработок в интернете без вложений",
                "Скидки до 90% только сегодня",
                "Казино онлайн, гарантированный выигрыш",
                # Срочные
                "Срочно нужна помощь, случилась авария!",
                "Пожар! Вызывайте скорую!",
                "Код красный, требуется немедленное вмешательство",
                "Проблема, ответь как можно быстрее",
                "Скорая помощь нужна срочно",
                # Рабочие
                "Завтра дедлайн по проекту",
                "Начальник вызывает на совещание",
                "Клиент прислал правки в договор",
                "Отчет нужно сдать до вечера",
                "Рабочая встреча переносится",
                # Семейные
                "Мама просила позвонить",
                "Жена сказала купить продукты",
                "Сын получил пятерку в школе",
                "Брат приедет в гости на выходные",
                "Дочка заболела, нужно в поликлинику",
                # Личные
                "Как дела? Что делаешь?",
                "Привет! Давно не виделись",
                "Пойдем гулять вечером?",
                "С днем рождения!",
                "Спокойной ночи",
                # Другие
                "Сегодня отличная погода",
                "Новости: курс доллара вырос",
                "Смотри, какой смешной мем",
                "Вкусный рецепт на ужин",
                "Интересная статья по психологии"
            ],
            'category': [
                'spam', 'spam', 'spam', 'spam', 'spam',
                'urgent', 'urgent', 'urgent', 'urgent', 'urgent',
                'work', 'work', 'work', 'work', 'work',
                'family', 'family', 'family', 'family', 'family',
                'personal', 'personal', 'personal', 'personal', 'personal',
                'other', 'other', 'other', 'other', 'other'
            ]
        }
        return pd.DataFrame(samples)

    def train(self):
        """Обучает модель с улучшенной обработкой ошибок"""
        try:
            # Загружаем данные
            df = self.load_training_data_from_db()

            if len(df) < 5:
                logger.warning("Слишком мало данных для обучения. Нужно минимум 5 примеров.")
                return None

            # Подготавливаем данные
            logger.info("Предобработка текстов...")
            X = [self.preprocess_text(t) for t in df['text'].tolist()]
            y = df['category'].tolist()

            # Проверяем, что все категории представлены
            unique_categories = set(y)
            logger.info(f"Категории в данных: {unique_categories}")

            # Разделяем на обучающую и тестовую выборки
            if len(df) >= 10:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            else:
                # Если данных мало, используем все для обучения
                X_train, y_train = X, y
                X_test, y_test = X, y

            # Создаем пайплайн
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=1000,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=1.0
                )),
                ('clf', LogisticRegression(
                    max_iter=1000,
                    random_state=42
                ))
            ])

            # Обучаем
            logger.info("Обучение модели...")
            pipeline.fit(X_train, y_train)

            self.model = pipeline

            # Оцениваем качество
            accuracy = pipeline.score(X_test, y_test)
            logger.info(f"✅ Модель обучена! Точность: {accuracy:.2%}")

            # Сохраняем модель
            self.save_model()

            return {
                'accuracy': accuracy,
                'samples_count': len(df),
                'categories': list(unique_categories)
            }

        except Exception as e:
            logger.error(f"❌ Ошибка при обучении: {e}")
            import traceback
            traceback.print_exc()
            return None

    def predict(self, text):
        """Предсказывает категорию сообщения"""
        if self.model is None:
            if not self.load_model():
                return {
                    'category': 'other',
                    'confidence': 0.5,
                    'score': 5,
                    'probabilities': {}
                }

        try:
            # Предобработка
            processed = self.preprocess_text(text)

            # Предсказание
            probs = self.model.predict_proba([processed])[0]
            pred = self.model.predict([processed])[0]
            confidence = max(probs)

            # Вычисляем оценку срочности
            score = self._calculate_urgency_score(pred, text, confidence)

            return {
                'category': pred,
                'confidence': confidence,
                'score': score,
                'probabilities': dict(zip(self.model.classes_, probs))
            }

        except Exception as e:
            logger.error(f"Ошибка при предсказании: {e}")
            return {
                'category': 'other',
                'confidence': 0.5,
                'score': 5,
                'probabilities': {}
            }

    def _calculate_urgency_score(self, category, text, confidence):
        """Вычисляет оценку срочности"""
        base_scores = {
            'urgent': 15,
            'family': 10,
            'work': 8,
            'personal': 6,
            'other': 4,
            'spam': 1
        }

        score = base_scores.get(category, 5)

        # Корректируем на уверенность
        score = int(score * (0.5 + confidence * 0.5))

        # Проверяем ключевые слова срочности
        urgent_words = ['срочно', 'немедленно', 'пожар', 'авария', 'help', 'urgent']
        text_lower = text.lower()
        for word in urgent_words:
            if word in text_lower:
                score += 3

        return min(20, max(0, score))

    def save_model(self):
        """Сохраняет модель"""
        try:
            joblib.dump(self.model, self.model_path)
            logger.info(f"✅ Модель сохранена в {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка при сохранении модели: {e}")
            return False

    def load_model(self):
        """Загружает модель"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info(f"✅ Модель загружена из {self.model_path}")
                return True
            else:
                logger.warning("Файл модели не найден")
                return False
        except Exception as e:
            logger.error(f"❌ Ошибка при загрузке модели: {e}")
            return False

    def get_advice(self, category, score, confidence):
        """Генерирует совет"""
        base_advice = {
            'spam': "🔴 СПАМ! Можно удалить.",
            'urgent': "⚠️ КРИТИЧНО! Ответьте немедленно!",
            'work': "💼 Рабочее. Ответьте в рабочее время.",
            'family': "💚 Семейное. Не игнорируйте.",
            'personal': "👤 Личное. Можно ответить, когда удобно.",
            'other': "🔵 Обычное сообщение."
        }

        advice = base_advice.get(category, "📨 Обычное сообщение")

        if score > 15:
            advice = "⚠️ КРИТИЧНО! Ответьте немедленно!"
        elif score < 3:
            advice = "🔴 Спам или неважное сообщение"

        if confidence < 0.6:
            advice += f" (уверенность: {int(confidence * 100)}%)"

        return advice


# Создаем экземпляр модели
my_model = MySpamModel()
