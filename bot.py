import asyncio
import logging
import re
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.enums import ParseMode
from my_spam_model import my_model
import os

# ========== НАСТРОЙКИ ==========
API_TOKEN = os.getenv('BOT_TOKEN', 'YOUR TOKEN')
# ===============================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

bot = Bot(token=API_TOKEN)
dp = Dispatcher()


# Инициализация базы данных
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (user_id INTEGER PRIMARY KEY, 
                  username TEXT,
                  first_name TEXT,
                  sleep_mode INTEGER DEFAULT 0,
                  wake_time TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  message_text TEXT,
                  sender TEXT,
                  category TEXT,
                  urgency_score INTEGER,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()


init_db()


# Добавьте в начало файла, после init_db()
def init_training_db():
    """Инициализация базы данных для обучающих примеров"""
    conn = sqlite3.connect('training_data.db')
    c = conn.cursor()

    # Создаем таблицу для обучающих примеров с колонкой needs_review
    c.execute('''CREATE TABLE IF NOT EXISTS training_samples
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  text TEXT NOT NULL,
                  category TEXT DEFAULT 'unlabeled',
                  marked_by INTEGER,
                  needs_review INTEGER DEFAULT 1,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    # Проверяем, есть ли уже таблица и нужно ли добавить колонку
    c.execute("PRAGMA table_info(training_samples)")
    columns = [column[1] for column in c.fetchall()]

    # Если колонки needs_review нет, добавляем её
    if 'needs_review' not in columns:
        try:
            c.execute("ALTER TABLE training_samples ADD COLUMN needs_review INTEGER DEFAULT 1")
            print("✅ Добавлена колонка needs_review")
        except Exception as e:
            print(f"❌ Ошибка при добавлении колонки: {e}")

    # Создаем индексы (с проверкой существования)
    try:
        c.execute('''CREATE INDEX IF NOT EXISTS idx_training_needs_review 
                     ON training_samples(needs_review)''')
    except Exception as e:
        print(f"⚠️ Не удалось создать индекс needs_review: {e}")

    try:
        c.execute('''CREATE INDEX IF NOT EXISTS idx_training_marked_by 
                     ON training_samples(marked_by)''')
    except Exception as e:
        print(f"⚠️ Не удалось создать индекс marked_by: {e}")

    conn.commit()
    conn.close()
    print("✅ Таблица training_samples проверена и обновлена")


# Вызовите эту функцию при запуске
init_training_db()

# Функции для работы с БД
def get_user(user_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    user = c.fetchone()
    conn.close()
    return user


def create_user(user_id, username, first_name):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO users (user_id, username, first_name) VALUES (?, ?, ?)",
              (user_id, username, first_name))
    conn.commit()
    conn.close()


def set_sleep_mode(user_id, hours):
    wake_time = datetime.now() + timedelta(hours=hours)
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("UPDATE users SET sleep_mode = 1, wake_time = ? WHERE user_id = ?",
              (wake_time.isoformat(), user_id))
    conn.commit()
    conn.close()
    return wake_time


def disable_sleep_mode(user_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("UPDATE users SET sleep_mode = 0, wake_time = NULL WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()


def save_message(user_id, message_text, sender, category, urgency_score):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("""INSERT INTO messages (user_id, message_text, sender, category, urgency_score)
                 VALUES (?, ?, ?, ?, ?)""",
              (user_id, message_text[:200], sender, category, urgency_score))
    conn.commit()
    conn.close()


# Добавьте новые функции для работы с обучающей выборкой
def save_to_training(user_id: int, text: str, category: str = None):
    """
    Сохраняет сообщение в базу для обучения
    Если category=None, сохраняем для ручной разметки
    """
    try:
        conn = sqlite3.connect('training_data.db')
        c = conn.cursor()

        # Проверяем, не сохраняли ли мы уже это сообщение
        c.execute("SELECT id FROM training_samples WHERE text = ? AND marked_by = ?",
                  (text[:500], user_id))
        if not c.fetchone():
            c.execute("""INSERT INTO training_samples (text, category, marked_by, needs_review) 
                         VALUES (?, ?, ?, ?)""",
                      (text[:500], category if category else 'unlabeled', user_id, 1 if not category else 0))
            conn.commit()
            logger.info(f"✅ Сообщение сохранено для обучения (категория: {category or 'не размечено'})")
        conn.close()
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка при сохранении в training_data: {e}")
        return False


def get_unlabeled_count(user_id: int) -> int:
    """Возвращает количество неразмеченных сообщений"""
    try:
        conn = sqlite3.connect('training_data.db')
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM training_samples WHERE marked_by = ? AND needs_review = 1",
                  (user_id,))
        count = c.fetchone()[0]
        conn.close()
        return count
    except:
        return 0


def get_next_unlabeled(user_id: int):
    """Получает следующее неразмеченное сообщение для разметки"""
    try:
        conn = sqlite3.connect('training_data.db')
        c = conn.cursor()
        c.execute("""SELECT id, text FROM training_samples 
                     WHERE marked_by = ? AND needs_review = 1 
                     ORDER BY created_at ASC LIMIT 1""", (user_id,))
        result = c.fetchone()
        conn.close()
        return result if result else None
    except:
        return None


def update_training_category(sample_id: int, category: str):
    """Обновляет категорию размеченного сообщения"""
    try:
        conn = sqlite3.connect('training_data.db')
        c = conn.cursor()
        c.execute("""UPDATE training_samples 
                     SET category = ?, needs_review = 0 
                     WHERE id = ?""", (category, sample_id))
        conn.commit()
        conn.close()
        return True
    except:
        return False

def get_digest(user_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    yesterday = (datetime.now() - timedelta(days=1)).isoformat()
    c.execute("""SELECT category, COUNT(*) FROM messages 
                 WHERE user_id = ? AND created_at > ? 
                 GROUP BY category""", (user_id, yesterday))
    stats = c.fetchall()
    conn.close()
    return stats


# Анализ сообщений
def analyze_message(text: str, sender: str = "unknown") -> dict:
    """
    Анализирует сообщение с использованием вашей собственной модели
    """
    # Используем свою модель для предсказания
    prediction = my_model.predict(text)

    # Получаем совет
    advice = my_model.get_advice(
        prediction['category'],
        prediction['score'],
        prediction['confidence']
    )

    return {
        "score": prediction['score'],
        "category": prediction['category'],
        "advice": advice,
        "confidence": prediction['confidence']
    }


@dp.message(Command("train_model"))
async def cmd_train_model(message: Message):
    """
    Запускает обучение модели на собранных данных
    """
    await message.answer("🔄 Начинаю обучение модели. Это может занять несколько минут...")

    try:
        # Обучаем модель
        results = my_model.train()

        if results is None:
            await message.answer(
                "❌ Не удалось обучить модель.\n\n"
                "Возможные причины:\n"
                "• Нет размеченных данных (используйте /mark)\n"
                "• Слишком мало примеров (нужно минимум 5)\n"
                "• Ошибка в данных"
            )
            return

        # Формируем отчет
        report = f"""✅ Модель успешно обучена!

📊 Метрики качества:
• Точность: {results['accuracy']:.2%}
• Примеров в обучении: {results['samples_count']}
• Категории: {', '.join(results['categories'])}

Модель сохранена и готова к использованию!"""

        await message.answer(report, parse_mode="HTML")

    except Exception as e:
        await message.answer(f"❌ Ошибка при обучении: {str(e)}")


# Добавьте команду для просмотра статистики модели
@dp.message(Command("model_stats"))
async def cmd_model_stats(message: Message):
    """
    Показывает статистику модели
    """
    # Получаем данные из БД
    conn = sqlite3.connect('training_data.db')
    c = conn.cursor()
    c.execute("SELECT category, COUNT(*) FROM training_samples GROUP BY category")
    stats = c.fetchall()
    conn.close()

    if not stats:
        await message.answer("📭 Нет размеченных данных для обучения. Используйте /mark для разметки сообщений.")
        return

    text = "📊 <b>Статистика обучающей выборки:</b>\n\n"
    total = sum(count for _, count in stats)

    for cat, count in stats:
        percent = (count / total) * 100
        text += f"• {cat}: {count} ({percent:.1f}%)\n"

    text += f"\n<b>Всего:</b> {total} размеченных сообщений"
    text += "\n\n💡 Чем больше данных, тем точнее будет модель!"

    await message.answer(text, parse_mode="HTML")


# Добавьте команду для экспорта данных
@dp.message(Command("export_data"))
async def cmd_export_data(message: Message):
    """
    Экспортирует размеченные данные в CSV
    """
    conn = sqlite3.connect('training_data.db')
    df = pd.read_sql_query("SELECT text, category, created_at FROM training_samples", conn)
    conn.close()

    if len(df) == 0:
        await message.answer("📭 Нет данных для экспорта")
        return

    # Сохраняем в CSV
    filename = f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False, encoding='utf-8')

    # Отправляем файл
    with open(filename, 'rb') as f:
        await message.answer_document(
            types.BufferedInputFile(f.read(), filename=filename),
            caption=f"📁 Экспортировано {len(df)} размеченных сообщений"
        )

    # Удаляем временный файл
    os.remove(filename)
# Команды бота
@dp.message(Command("start"))
async def cmd_start(message: Message):
    user = message.from_user
    create_user(user.id, user.username, user.first_name)

    welcome_text = f"""
👋 Привет, <b>{user.first_name}</b>!

Я твой <b>цифровой секретарь</b> — помогаю фильтровать шум и оставлять только важное.

📌 <b>Что я умею:</b>
• Анализировать пересланные сообщения
• Режим «Не беспокоить» с автоответом
• Ежедневный дайджест пропущенного
• <b>НОВОЕ:</b> Самообучаюсь на ваших сообщениях!

📎 <b>Как пользоваться:</b>
Просто пересылай мне любые сообщения — я скажу, насколько они важны.
Неразмеченные сообщения автоматически сохраняются для обучения.

⚙️ <b>Команды:</b>
/sleep 2 — уйти в сон на 2 часа
/wake — проснуться
/digest — дайджест за вчера
/review — разметить неразмеченные сообщения
/train_model — обучить модель на размеченных данных
/stats_training — статистика обучающей выборки
/help — все команды
"""
    await message.answer(welcome_text, parse_mode="HTML")


@dp.message(Command("help"))
async def cmd_help(message: Message):
    help_text = """
🔍 <b>Все команды:</b>

/sleep N — уйти в режим невидимки на N часов (например: /sleep 3)
/wake — выключить режим сна
/digest — получить дайджест сообщений за вчера
/stats — статистика по категориям
/about — о боте

💡 <b>Совет:</b> Пересылай мне сообщения из любых чатов, чтобы я научился лучше понимать твои приоритеты.
"""
    await message.answer(help_text, parse_mode="HTML")


@dp.message(Command("about"))
async def cmd_about(message: Message):
    about_text = """
🧠 <b>Reality Filter Bot</b> v1.0

AI-ассистент для фильтрации информационного шума. Помогает не сойти с ума в мире уведомлений.

Особенности:
• Автономный анализ текста
• Режим глубокого сна
• Приватность (данные хранятся локально)

Создан с ❤️ для тех, кто ценит тишину.
"""
    await message.answer(about_text, parse_mode="HTML")


@dp.message(Command("sleep"))
async def cmd_sleep(message: Message):
    args = message.text.split()
    if len(args) > 1:
        try:
            hours = float(args[1])
            if hours > 24:
                await message.answer("⏰ Максимум 24 часа. Укажите меньшее значение.")
                return

            wake_time = set_sleep_mode(message.from_user.id, hours)
            await message.answer(
                f"😴 <b>Режим сна активирован</b> на {hours} ч.\n"
                f"Проснусь: {wake_time.strftime('%H:%M %d.%m')}\n\n"
                f"Все входящие получат автоответ.",
                parse_mode="HTML"
            )
        except ValueError:
            await message.answer("❌ Укажите число часов, например: /sleep 2")
    else:
        await message.answer("❌ Укажите время, например: /sleep 2 (часа)")


@dp.message(Command("wake"))
async def cmd_wake(message: Message):
    user = get_user(message.from_user.id)
    if user and user[3] == 1:  # sleep_mode = 1
        disable_sleep_mode(message.from_user.id)
        await message.answer("👋 <b>Я проснулся!</b> Снова на связи.", parse_mode="HTML")
    else:
        await message.answer("✅ Я и не спал. Работаю в штатном режиме.")


@dp.message(Command("digest"))
async def cmd_digest(message: Message):
    stats = get_digest(message.from_user.id)

    if not stats:
        await message.answer("📭 За вчера не было проанализированных сообщений.")
        return

    digest_text = "📊 <b>Дайджест за вчера:</b>\n\n"
    total = 0

    category_names = {
        "urgent": "⚠️ Срочные",
        "work": "💼 Рабочие",
        "family": "👨‍👩‍👧 Семья",
        "spam": "📛 Спам",
        "other": "📨 Прочие"
    }

    for cat, count in stats:
        name = category_names.get(cat, cat)
        digest_text += f"{name}: {count}\n"
        total += count

    digest_text += f"\n<b>Всего:</b> {total} сообщений"

    if total > 20:
        digest_text += "\n\n💡 Многовато шума. Попробуй режим /sleep почаще."

    await message.answer(digest_text, parse_mode="HTML")


@dp.message(Command("stats"))
async def cmd_stats(message: Message):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("""SELECT category, COUNT(*) FROM messages 
                 WHERE user_id = ? 
                 GROUP BY category ORDER BY COUNT(*) DESC""",
              (message.from_user.id,))
    stats = c.fetchall()
    conn.close()

    if not stats:
        await message.answer("📭 Пока нет статистики. Начни пересылать мне сообщения!")
        return

    text = "📈 <b>Общая статистика:</b>\n\n"
    total = sum(count for _, count in stats)

    emoji_map = {
        "urgent": "⚠️", "work": "💼", "family": "👪",
        "spam": "🚫", "other": "📄"
    }

    for cat, count in stats:
        emoji = emoji_map.get(cat, "📌")
        percent = (count / total) * 100
        text += f"{emoji} {cat.capitalize()}: {count} ({percent:.1f}%)\n"

    await message.answer(text, parse_mode="HTML")


# Обработка пересланных сообщений
@dp.message(lambda message: message.forward_from or message.forward_sender_name or message.forward_from_chat)
async def handle_forwarded(message: Message):
    """Обработка пересланных сообщений с автосохранением для обучения"""

    # Определяем отправителя
    if message.forward_from:
        sender = message.forward_from.full_name
        if message.forward_from.username:
            sender += f" (@{message.forward_from.username})"
    elif message.forward_from_chat:
        sender = f"Канал: {message.forward_from_chat.title}"
    else:
        sender = message.forward_sender_name or "Неизвестный отправитель"

    # Текст сообщения
    text = message.text or message.caption or ""

    if not text:
        await message.answer("❌ Не могу проанализировать сообщение без текста.")
        return

    # Анализируем сообщение
    analysis = analyze_message(text, sender)

    # Сохраняем в основную БД
    save_message(message.from_user.id, text, sender, analysis["category"], analysis["score"])

    # АВТОМАТИЧЕСКИ сохраняем в обучающую выборку
    # Если уверенность модели низкая (< 0.7), помечаем для ручной проверки
    if analysis.get('confidence', 0) < 0.7:
        save_to_training(message.from_user.id, text)
        needs_review = True
    else:
        # Если модель уверена, сохраняем с предсказанной категорией
        save_to_training(message.from_user.id, text, analysis["category"])
        needs_review = False

    # Проверяем режим сна
    user = get_user(message.from_user.id)
    if user and user[3] == 1:  # sleep_mode = 1
        wake_time = datetime.fromisoformat(user[4])
        if datetime.now() < wake_time:
            await message.answer(
                f"🤖 <b>Автоответ:</b> Пользователь в режиме сна до {wake_time.strftime('%H:%M')}.\n"
                f"Сообщение от <i>{sender}</i> будет доставлено после пробуждения.",
                parse_mode="HTML"
            )
            return

    # Формируем ответ
    response = f"📨 <b>От:</b> {sender}\n\n"
    response += f"📝 <b>Текст:</b> {text[:200]}"
    if len(text) > 200:
        response += "..."

    response += f"\n\n{analysis['advice']}"

    # Добавляем информацию о необходимости разметки
    unlabeled = get_unlabeled_count(message.from_user.id)
    if needs_review:
        response += f"\n\n❓ <b>Нужна проверка:</b> Модель не уверена в категории."

    if unlabeled > 0:
        response += f"\n📊 <b>Ожидают разметки:</b> {unlabeled} сообщений. Используйте /review для разметки."

    if analysis['score'] > 12:
        response += "\n\n⚡ <b>Рекомендуется ответить как можно скорее.</b>"

    await message.answer(response, parse_mode="HTML")


@dp.message(Command("review"))
async def cmd_review(message: Message):
    """
    Начинает процесс разметки неразмеченных сообщений
    """
    # Получаем следующее неразмеченное сообщение
    sample = get_next_unlabeled(message.from_user.id)

    if not sample:
        await message.answer("✅ Нет сообщений для разметки! Все размечены.")
        return

    sample_id, text = sample

    # Сохраняем ID текущего сообщения в контекст
    # Для простоты используем глобальную переменную или словарь
    # В продакшене лучше использовать Redis или БД
    if not hasattr(message.bot, 'review_context'):
        message.bot.review_context = {}

    message.bot.review_context[message.from_user.id] = sample_id

    # Создаем клавиатуру для выбора категории
    keyboard = [
        [types.InlineKeyboardButton(text="🔴 Спам", callback_data="review_spam")],
        [types.InlineKeyboardButton(text="⚠️ Срочно", callback_data="review_urgent")],
        [types.InlineKeyboardButton(text="💼 Работа", callback_data="review_work")],
        [types.InlineKeyboardButton(text="💚 Семья", callback_data="review_family")],
        [types.InlineKeyboardButton(text="👤 Личное", callback_data="review_personal")],
        [types.InlineKeyboardButton(text="📨 Другое", callback_data="review_other")],
        [types.InlineKeyboardButton(text="⏭ Пропустить", callback_data="review_skip")],
    ]
    markup = types.InlineKeyboardMarkup(inline_keyboard=keyboard)

    await message.answer(
        f"📝 <b>Разметка сообщения ({get_unlabeled_count(message.from_user.id)} осталось):</b>\n\n"
        f"Текст: {text}\n\n"
        f"Выберите категорию:",
        parse_mode="HTML",
        reply_markup=markup
    )


@dp.callback_query(lambda c: c.data and c.data.startswith('review_'))
async def process_review_callback(callback_query: types.CallbackQuery):
    """Обработка выбора категории при разметке"""
    user_id = callback_query.from_user.id

    # Получаем ID сообщения из контекста
    if not hasattr(callback_query.bot, 'review_context') or user_id not in callback_query.bot.review_context:
        await callback_query.answer("❌ Сессия разметки истекла. Начните заново с /review")
        await callback_query.message.delete()
        return

    sample_id = callback_query.bot.review_context[user_id]
    action = callback_query.data.replace('review_', '')

    if action == 'skip':
        # Просто пропускаем, не меняя статус
        await callback_query.answer("⏭ Пропущено")
        await callback_query.message.delete()

        # Показываем следующее сообщение
        next_sample = get_next_unlabeled(user_id)
        if next_sample:
            # Создаем новую команду review
            await cmd_review(callback_query.message)
        else:
            await callback_query.message.answer("✅ Все сообщения размечены!")
        return

    # Маппинг на английские категории
    category_map = {
        'spam': 'spam',
        'urgent': 'urgent',
        'work': 'work',
        'family': 'family',
        'personal': 'personal',
        'other': 'other'
    }

    if action in category_map:
        # Обновляем категорию
        if update_training_category(sample_id, category_map[action]):
            await callback_query.answer(f"✅ Сообщение помечено как {action}")

            # Удаляем сообщение с разметкой
            await callback_query.message.delete()

            # Показываем следующее сообщение
            next_sample = get_next_unlabeled(user_id)
            if next_sample:
                # Сохраняем новый ID
                callback_query.bot.review_context[user_id] = next_sample[0]

                # Создаем новую клавиатуру
                keyboard = [
                    [types.InlineKeyboardButton(text="🔴 Спам", callback_data="review_spam")],
                    [types.InlineKeyboardButton(text="⚠️ Срочно", callback_data="review_urgent")],
                    [types.InlineKeyboardButton(text="💼 Работа", callback_data="review_work")],
                    [types.InlineKeyboardButton(text="💚 Семья", callback_data="review_family")],
                    [types.InlineKeyboardButton(text="👤 Личное", callback_data="review_personal")],
                    [types.InlineKeyboardButton(text="📨 Другое", callback_data="review_other")],
                    [types.InlineKeyboardButton(text="⏭ Пропустить", callback_data="review_skip")],
                ]
                markup = types.InlineKeyboardMarkup(inline_keyboard=keyboard)

                await callback_query.message.answer(
                    f"📝 <b>Следующее сообщение ({get_unlabeled_count(user_id)} осталось):</b>\n\n"
                    f"Текст: {next_sample[1]}\n\n"
                    f"Выберите категорию:",
                    parse_mode="HTML",
                    reply_markup=markup
                )
            else:
                await callback_query.message.answer(
                    "✅ Все сообщения размечены! Можете обучить модель командой /train_model")
        else:
            await callback_query.answer("❌ Ошибка при сохранении")


@dp.message(Command("stats_training"))
async def cmd_stats_training(message: Message):
    """Показывает статистику обучающей выборки"""
    try:
        conn = sqlite3.connect('training_data.db')
        c = conn.cursor()

        # Общая статистика
        c.execute("SELECT COUNT(*) FROM training_samples WHERE marked_by = ?", (message.from_user.id,))
        total = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM training_samples WHERE marked_by = ? AND needs_review = 1",
                  (message.from_user.id,))
        unlabeled = c.fetchone()[0]

        # Статистика по категориям
        c.execute("""SELECT category, COUNT(*) FROM training_samples 
                     WHERE marked_by = ? AND needs_review = 0
                     GROUP BY category""", (message.from_user.id,))
        stats = c.fetchall()

        conn.close()

        text = f"📊 <b>Статистика обучающей выборки:</b>\n\n"
        text += f"📝 Всего сообщений: {total}\n"
        text += f"❓ Не размечено: {unlabeled}\n"
        text += f"✅ Размечено: {total - unlabeled}\n\n"

        if stats:
            text += "<b>По категориям:</b>\n"
            for cat, count in stats:
                emoji = {
                    'spam': '🔴',
                    'urgent': '⚠️',
                    'work': '💼',
                    'family': '💚',
                    'personal': '👤',
                    'other': '📨'
                }.get(cat, '📌')
                text += f"{emoji} {cat}: {count}\n"

        text += f"\n💡 Используйте /review для разметки неразмеченных сообщений"

        await message.answer(text, parse_mode="HTML")

    except Exception as e:
        await message.answer(f"❌ Ошибка: {e}")


@dp.message(Command("auto_mark"))
async def cmd_auto_mark(message: Message):
    """Включает/выключает автоматическое добавление в обучающую выборку"""
    # Сохраняем настройку в БД пользователя
    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    # Добавляем колонку если её нет
    try:
        c.execute("ALTER TABLE users ADD COLUMN auto_mark INTEGER DEFAULT 1")
    except:
        pass

    # Получаем текущее значение
    c.execute("SELECT auto_mark FROM users WHERE user_id = ?", (message.from_user.id,))
    result = c.fetchone()

    if result and result[0] == 1:
        new_value = 0
        status = "выключен"
    else:
        new_value = 1
        status = "включен"

    c.execute("UPDATE users SET auto_mark = ? WHERE user_id = ?", (new_value, message.from_user.id))
    conn.commit()
    conn.close()

    await message.answer(
        f"🤖 <b>Автосбор данных {status}!</b>\n\n"
        f"Когда режим включен, все пересланные сообщения автоматически "
        f"добавляются в обучающую выборку."
    )

# Обработка обычных сообщений (не пересланных)
@dp.message()
async def handle_text(message: Message):
    if message.text and not message.text.startswith('/'):
        await message.answer(
            "📎 Чтобы я проанализировал сообщение, <b>перешли его мне</b>.\n\n"
            "Нажми на сообщение → «Переслать» → выбери меня.",
            parse_mode="HTML"
        )


# Добавьте в основной файл бота





    # Получаем текст сообщения
    replied = message.reply_to_message
    text = replied.text or replied.caption or ""

    if not text:
        await message.answer("❌ Нет текста для разметки")
        return

    # Сохраняем в отдельную таблицу для обучения
    conn = sqlite3.connect('training_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS training_samples
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  text TEXT,
                  category TEXT,
                  marked_by INTEGER,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    c.execute("INSERT INTO training_samples (text, category, marked_by) VALUES (?, ?, ?)",
              (text[:500], message.from_user.id))
    conn.commit()
    conn.close()

    # await message.answer(f"✅ Сообщение помечено как '' и добавлено в обучающую выборку")

# Запуск бота
async def main():
    logger.info("Бот запускается...")
    await dp.start_polling(bot)

# Пример добавления дополнительных признаков
def extract_features(text):
    features = {
        'length': len(text),
        'has_link': 1 if re.search(r'http[s]?://', text) else 0,
        'has_exclamation': text.count('!'),
        'has_question': text.count('?'),
        'caps_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
        'word_count': len(text.split())
    }
    return features


def analyze_message(text: str, sender: str = "unknown") -> dict:
    """
    Анализирует сообщение с использованием вашей собственной модели
    """
    # Используем свою модель для предсказания
    prediction = my_model.predict(text)

    # Получаем совет
    advice = my_model.get_advice(
        prediction['category'],
        prediction['score'],
        prediction['confidence']
    )

    # Добавляем уровень уверенности
    confidence_level = "высокая" if prediction['confidence'] > 0.8 else "средняя" if prediction[
                                                                                         'confidence'] > 0.6 else "низкая"

    return {
        "score": prediction['score'],
        "category": prediction['category'],
        "advice": advice,
        "confidence": prediction['confidence'],
        "confidence_level": confidence_level,
        "probabilities": prediction.get('probabilities', {})
    }

if __name__ == '__main__':
    asyncio.run(main())
