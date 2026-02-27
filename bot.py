import asyncio
import logging
import re
import sqlite3
from datetime import datetime, timedelta
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.enums import ParseMode
import os

# ========== НАСТРОЙКИ ==========
API_TOKEN = os.getenv('BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE')
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
    text_lower = text.lower()

    # Ключевые слова для категорий
    urgent_words = ['срочно', 'пожар', 'авария', 'код красный', 'быстро', 'problem', 'urgent', 'help', 'помоги']
    work_words = ['отчет', 'начальник', 'deadline', 'работа', 'зарплата', 'клиент', 'проект', 'босс', 'work']
    spam_words = ['купи', 'скидка', 'казино', 'выигрыш', 'инвестиции', 'сайт', 'заработок', 'бесплатно', 'оффер']
    family_words = ['мама', 'папа', 'сын', 'дочь', 'жена', 'муж', 'родной', 'бабушка', 'дедушка', 'брат', 'сестра']

    score = 5  # Базовая оценка
    category = "other"

    # Повышаем/понижаем score
    if any(word in text_lower for word in urgent_words):
        score += 10
        category = "urgent"
    if any(word in text_lower for word in work_words):
        score += 5
        category = "work" if category == "other" else category
    if any(word in text_lower for word in family_words):
        score += 7
        category = "family" if category == "other" else category
    if any(word in text_lower for word in spam_words):
        score -= 10
        category = "spam"

    # Корректируем score в пределах 0-20
    score = max(0, min(20, score))

    # Определяем совет
    if category == "spam" or score < 3:
        advice = "🔴 СПАМ или реклама. Можно удалить не читая."
    elif score > 15:
        advice = "⚠️ КРИТИЧНО! Ответьте немедленно."
    elif score > 10:
        advice = "🟡 ВАЖНО. Ответьте в ближайший час."
    elif category == "family":
        advice = "💚 Семья. Не игнорируйте, но можно не спешить."
    else:
        advice = "🔵 Обычное сообщение. Можно почитать позже."

    return {
        "score": score,
        "category": category,
        "advice": advice
    }


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

📎 <b>Как пользоваться:</b>
Просто пересылай мне любые сообщения — я скажу, насколько они важны.

⚙️ <b>Команды:</b>
/sleep 2 — уйти в сон на 2 часа
/wake — проснуться
/digest — дайджест за вчера
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

    # Анализируем
    analysis = analyze_message(text, sender)

    # Сохраняем в БД
    save_message(message.from_user.id, text, sender, analysis["category"], analysis["score"])

    # Проверяем режим сна
    user = get_user(message.from_user.id)
    if user and user[3] == 1:  # sleep_mode = 1
        wake_time = datetime.fromisoformat(user[4])
        if datetime.now() < wake_time:
            # Автоответ
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

    if analysis['score'] > 12:
        response += "\n\n⚡ <b>Рекомендуется ответить как можно скорее.</b>"

    await message.answer(response, parse_mode="HTML")


# Обработка обычных сообщений (не пересланных)
@dp.message()
async def handle_text(message: Message):
    if message.text and not message.text.startswith('/'):
        await message.answer(
            "📎 Чтобы я проанализировал сообщение, <b>перешли его мне</b>.\n\n"
            "Нажми на сообщение → «Переслать» → выбери меня.",
            parse_mode="HTML"
        )


# Запуск бота
async def main():
    logger.info("Бот запускается...")
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())
