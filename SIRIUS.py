import sqlite3
from datetime import datetime, timezone
import logging
import numpy as np
from typing import List, Dict, Set
import re

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext, CallbackQueryHandler

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telethon import TelegramClient, events
from telethon.tl.types import Message as TelethonMessage

from sentence_transformers import SentenceTransformer


class Config:
    # Токен вашего Telegram-бота (получить у @BotFather)
    BOT_TOKEN = "1234567890abcdef"

    # API ID и API Hash вашего Telegram-приложения (получить на my.telegram.org)
    API_ID = 123456789
    API_HASH = "123abc"

    # Имя файла сессии для Telethon
    SESSION_NAME = "new_session"

    # Публичные каналы для мониторинга
    CHANNELS_TO_INDEX = ["channel_1", "channel_N"]

    # Название БД
    DATABASE_NAME = "search_data.db"

    # Трансформер
    MODEL_NAME = "all-MiniLM-L6-v2"

    # Схожесть текста
    SIMILARITY_THRESHOLD = 0.35

    # Максимальная длина сообщения
    MAX_MESSAGE_LENGTH = 4000


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

try:
    logger.info(f"Загрузка модели SentenceTransformer: {Config.MODEL_NAME}...")
    similarity_model = SentenceTransformer(Config.MODEL_NAME)
    logger.info("Модель успешно загружена.")
except Exception as e:
    logger.error(f"Ошибка загрузки модели SentenceTransformer: {e}")
    logger.error("Функционал определения схожести и поиска по смыслу будет недоступен.")
    similarity_model = None


class Database:
    def __init__(self, db_name: str):
        self.db_name = db_name
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_username TEXT NOT NULL,
            telegram_message_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            message_date TIMESTAMP NOT NULL,
            embedding BLOB,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(channel_username, telegram_message_id)
        )
        ''')
        self.conn.commit()

    def save_message(self, channel_username: str, msg_id: int, text: str, date: datetime,
                     embedding: np.ndarray = None) -> int | None:
        try:
            cursor = self.conn.cursor()
            embedding_blob = embedding.tobytes() if embedding is not None and similarity_model else None
            cursor.execute(
                "INSERT OR IGNORE INTO messages (channel_username, telegram_message_id, text, message_date, embedding) VALUES (?, ?, ?, ?, ?)",
                (channel_username.lower(), msg_id, text, date, embedding_blob)
            )
            self.conn.commit()
            return cursor.lastrowid if cursor.rowcount > 0 else None
        except sqlite3.Error as e:
            logger.error(f"Ошибка сохранения сообщения из {channel_username} в БД: {e}")
            return None

    def search_messages(self, query_text: str, query_terms: List[str], search_mode: str,  # Вместо is_phrase_search
                        similarity_query_embedding: np.ndarray = None, limit: int = 5) -> List[Dict]:
        results = []
        seen_message_ids = set()
        cursor = self.conn.cursor()

        if search_mode == "phrase" and query_text:
            sql_text_conditions = "lower(text) LIKE ?"
            text_params = ['%' + query_text.lower() + '%']
            logger.debug(f"Phrase search: condition='{sql_text_conditions}', params='{text_params}'")

            sql_query = f"""
                SELECT id, channel_username, telegram_message_id, text, message_date, embedding
                FROM messages WHERE {sql_text_conditions} ORDER BY message_date DESC LIMIT {limit}"""
            try:
                cursor.execute(sql_query, text_params)
                for row in cursor.fetchall():
                    if row[
                        0] not in seen_message_ids:
                        try:
                            msg_date_obj = datetime.fromisoformat(row[4]) if isinstance(row[4], str) else row[4]
                            results.append({
                                "db_id": row[0], "channel_username": row[1], "telegram_message_id": row[2],
                                "text": row[3], "message_date": msg_date_obj, "score": 1.0, "match_type": "phrase"
                            })
                            seen_message_ids.add(row[0])
                        except ValueError as ve:
                            logger.error(
                                f"DB Error (phrase search date conversion) for msg_id {row[0]}: {ve}, value: {row[4]}")
            except sqlite3.Error as e:
                logger.error(f"DB Error (phrase search): {e}")

        elif search_mode == "words" and query_terms:
            sql_text_conditions = " AND ".join([f"lower(text) LIKE ?" for _ in query_terms])
            text_params = ['%' + term + '%' for term in query_terms]
            logger.debug(f"All words search: condition='{sql_text_conditions}', params='{text_params}'")

            sql_query = f"""
                SELECT id, channel_username, telegram_message_id, text, message_date, embedding
                FROM messages WHERE {sql_text_conditions} ORDER BY message_date DESC LIMIT {limit}"""
            try:
                cursor.execute(sql_query, text_params)
                for row in cursor.fetchall():
                    if row[0] not in seen_message_ids:
                        try:
                            msg_date_obj = datetime.fromisoformat(row[4]) if isinstance(row[4], str) else row[4]
                            results.append({
                                "db_id": row[0], "channel_username": row[1], "telegram_message_id": row[2],
                                "text": row[3], "message_date": msg_date_obj, "score": 1.0, "match_type": "words"
                            })
                            seen_message_ids.add(row[0])
                        except ValueError as ve:
                            logger.error(
                                f"DB Error (words search date conversion) for msg_id {row[0]}: {ve}, value: {row[4]}")
            except sqlite3.Error as e:
                logger.error(f"DB Error (words search): {e}")

        needs_semantic_search = False
        if search_mode == "semantic":
            if not similarity_model or similarity_query_embedding is None:
                logger.warning("Semantic search requested, but model or query embedding is not available.")
            else:
                needs_semantic_search = True
        elif len(results) < limit and similarity_model and similarity_query_embedding is not None:
            needs_semantic_search = True

        if needs_semantic_search:
            logger.debug(f"Performing semantic search (mode: {search_mode}, current results: {len(results)}).")
            try:
                cursor.execute(
                    "SELECT id, channel_username, telegram_message_id, text, message_date, embedding FROM messages WHERE embedding IS NOT NULL"
                )
                potential_semantic_matches = []
                all_db_messages_for_semantic = cursor.fetchall()

                for row in all_db_messages_for_semantic:
                    if row[0] in seen_message_ids and search_mode != "semantic":
                        continue

                    msg_embedding_db = np.frombuffer(row[5], dtype=np.float32)
                    if np.linalg.norm(msg_embedding_db) == 0 or np.linalg.norm(similarity_query_embedding) == 0:
                        similarity_score = 0.0
                    else:
                        similarity_score = np.dot(similarity_query_embedding, msg_embedding_db) / (
                                np.linalg.norm(similarity_query_embedding) * np.linalg.norm(msg_embedding_db))
                    logger.info(f"Semantic check - Msg DB ID: {row[0]}, Score: {similarity_score:.4f}")

                    if similarity_score >= Config.SIMILARITY_THRESHOLD:
                        try:
                            msg_date_obj = datetime.fromisoformat(row[4]) if isinstance(row[4], str) else row[4]
                            is_new_match = True
                            if search_mode != "semantic":
                                for res_item in results:
                                    if res_item["db_id"] == row[0]:
                                        is_new_match = False
                                        break

                            if is_new_match:
                                potential_semantic_matches.append({
                                    "db_id": row[0], "channel_username": row[1], "telegram_message_id": row[2],
                                    "text": row[3], "message_date": msg_date_obj, "score": float(similarity_score),
                                    "match_type": "semantic"
                                })
                        except ValueError as ve:
                            logger.error(
                                f"DB Error (semantic date conversion) for msg_id {row[0]}: {ve}, value: {row[4]}")

                potential_semantic_matches.sort(key=lambda x: x["score"], reverse=True)

                if search_mode == "semantic":
                    results = potential_semantic_matches
                else:
                    for match in potential_semantic_matches:
                        if len(results) >= limit:
                            break
                        if match["db_id"] not in seen_message_ids:
                            results.append(match)
                            seen_message_ids.add(match["db_id"])
            except sqlite3.Error as e:
                logger.error(f"DB Error (semantic messages): {e}")

        def sort_key(item):
            match_type_priority = 0
            if item.get("match_type") == "phrase":
                match_type_priority = 2
            elif item.get("match_type") == "words":
                match_type_priority = 1

            score = item.get("score", 0.0)
            freshness = item["message_date"].timestamp() if item["message_date"] else 0
            return (match_type_priority, score, freshness)

        results.sort(key=sort_key, reverse=True)
        return results[:limit]

    def close(self):
        if self.conn:
            self.conn.close()
            logger.info("Соединение с БД закрыто.")

class NewsAggregatorBot:
    def __init__(self):
        self.db = Database(Config.DATABASE_NAME)
        self.ptb_application = Application.builder().token(Config.BOT_TOKEN).build()
        self.telethon_client = TelegramClient(Config.SESSION_NAME, Config.API_ID, Config.API_HASH)
        self.user_search_mode: Dict[int, str] = {}
        self.ptb_application.post_init = self.post_ptb_init
        self.ptb_application.post_shutdown = self.post_ptb_shutdown

        self._setup_ptb_handlers()
        self.channels_to_index_set: Set[str] = {ch.lower().replace("@", "") for ch in Config.CHANNELS_TO_INDEX}

    async def _parse_channel_history(self, channel_username: str, limit: int):
        if limit == 0:
            return
        logger.info(f"Парсинг истории для @{channel_username}, лимит: {limit} сообщений...")
        count = 0
        try:
            async for message in self.telethon_client.iter_messages(channel_username, limit=limit):
                if message.text:
                    cursor = self.db.conn.cursor()
                    cursor.execute("SELECT 1 FROM messages WHERE channel_username = ? AND telegram_message_id = ?",
                                   (channel_username.lower(), message.id))
                    if cursor.fetchone():
                        logger.debug(f"Сообщение {message.id} из @{channel_username} уже в БД (история).")
                        continue
                    msg_embedding = similarity_model.encode(message.text) if similarity_model else None
                    msg_date_utc = message.date.replace(tzinfo=timezone.utc) if message.date.tzinfo is None \
                        else message.date.astimezone(timezone.utc)

                    saved_id = self.db.save_message(
                        channel_username.lower(),
                        message.id,
                        message.text,
                        msg_date_utc,
                        msg_embedding
                    )
                    if saved_id:
                        count += 1
            logger.info(f"Завершён парсинг истории для @{channel_username}. Добавлено новых: {count}")
        except Exception as e:
            logger.error(f"Ошибка при парсинге истории @{channel_username}: {e}", exc_info=True)

    async def post_ptb_init(self, application: Application):
        logger.info("PTB инициализирован. Запуск Telethon клиента...")
        try:
            logger.info(f"Попытка запуска Telethon клиента с токеном бота...")
            await self.telethon_client.start(bot_token=Config.BOT_TOKEN)
            me = await self.telethon_client.get_me()
            logger.info(f"Telethon клиент успешно запущен как бот: {me.username if me.username else me.first_name}")

            logger.info(
                f"Telethon будет сохранять сообщения из каналов, указанных в CHANNELS_TO_INDEX: {self.channels_to_index_set}")
            if not self.channels_to_index_set:
                logger.warning("Список CHANNELS_TO_INDEX пуст! Бот не будет собирать сообщения.")

            self._setup_telethon_handlers()
        except Exception as e:
            logger.error(f"Ошибка запуска Telethon клиента: {e}", exc_info=True)
            raise RuntimeError(f"Telethon клиент не смог запуститься: {e}")

    async def post_ptb_shutdown(self, application: Application):
        logger.info("PTB завершает работу...")
        if self.telethon_client.is_connected():
            logger.info("Отключение Telethon клиента...")
            await self.telethon_client.disconnect()
            logger.info("Telethon клиент отключен.")
        self.db.close()

    def _setup_ptb_handlers(self):
        self.ptb_application.add_handler(CommandHandler("start", self.start_command))
        self.ptb_application.add_handler(
            CommandHandler("home", self.start_command))  # Кнопка "В начало" будет вызывать /start
        self.ptb_application.add_handler(CommandHandler("indexedchannels", self.indexed_channels_command))
        self.ptb_application.add_handler(CommandHandler("searchmode", self.search_mode_command))
        self.ptb_application.add_handler(
            CallbackQueryHandler(self.button_callback_handler))  # Один обработчик для всех кнопок
        self.ptb_application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_search_query))

    def _setup_telethon_handlers(self):
        @self.telethon_client.on(events.NewMessage())
        async def handle_new_channel_message(event: events.NewMessage.Event):
            message: TelethonMessage = event.message
            channel_username = None
            if hasattr(message.chat, 'username') and message.chat.username:
                channel_username = message.chat.username.lower()
            if not channel_username or channel_username not in self.channels_to_index_set:
                return
            if message.text:
                logger.info(f"Telethon: Новое сообщение из @{channel_username} (индексируемый): {message.text[:50]}...")
                msg_embedding = similarity_model.encode(message.text) if similarity_model else None
                msg_date_utc = message.date.replace(
                    tzinfo=timezone.utc) if message.date.tzinfo is None else message.date.astimezone(timezone.utc)

                self.db.save_message(
                    channel_username,
                    message.id,
                    message.text,
                    msg_date_utc,
                    msg_embedding
                )
    async def start_command(self, update: Update, context: CallbackContext, message_id_to_edit: int = None):
        user = update.effective_user
        if user:
            self.user_search_mode[user.id] = "words"

        text = (
            rf"Привет, {user.mention_html() if user else ''}! Я бот для поиска по каналам."
            "\n\nИспользуй команды:"
            "\n/indexedchannels - список каналов для поиска."
            "\n/searchmode - изменить режим поиска."
            f"\n\nТекущий режим: <b>{self._get_current_mode_display(user.id if user else None)}</b>."
            "\nОтправь мне текст для поиска."
        )
        keyboard = [
            [
                InlineKeyboardButton("По словам", callback_data="setmode_words"),
                InlineKeyboardButton("Фраза", callback_data="setmode_phrase"),
                InlineKeyboardButton("По смыслу", callback_data="setmode_semantic")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        if message_id_to_edit and update.callback_query:
            try:
                await context.bot.edit_message_text(
                    chat_id=update.callback_query.message.chat_id,
                    message_id=message_id_to_edit,
                    text=text,
                    reply_markup=reply_markup,
                    parse_mode=ParseMode.HTML
                )
            except Exception as e:
                logger.error(f"Ошибка редактирования сообщения для start_command: {e}")
                await context.bot.send_message(update.callback_query.message.chat_id, text, reply_markup=reply_markup,
                                               parse_mode=ParseMode.HTML)

        elif update.message:
            await update.message.reply_html(text, reply_markup=reply_markup)

    def _get_current_mode_display(self, user_id: int | None) -> str:
        if user_id is None: return "по словам (по умолч.)"
        mode = self.user_search_mode.get(user_id, "words")
        if mode == "words": return "по словам"
        if mode == "phrase": return "точная фраза"
        if mode == "semantic": return "по смыслу"
        return "неизвестно"

    async def search_mode_command(self, update: Update, context: CallbackContext):
        keyboard = [
            [
                InlineKeyboardButton("По словам", callback_data="setmode_words"),
                InlineKeyboardButton("Фраза", callback_data="setmode_phrase"),
                InlineKeyboardButton("По смыслу", callback_data="setmode_semantic")
            ],
            [InlineKeyboardButton("🏠 В начало", callback_data="gohome")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        current_mode_display = self._get_current_mode_display(update.effective_user.id)
        await update.message.reply_text(f"Выберите режим поиска. Текущий: {current_mode_display}",
                                        reply_markup=reply_markup)

    async def button_callback_handler(self, update: Update, context: CallbackContext):
        query = update.callback_query
        await query.answer()
        user_id = query.from_user.id

        action = query.data.split('_')[0]

        if query.data == "gohome":
            await self.start_command(update, context, message_id_to_edit=query.message.message_id)
            return

        if action == "setmode":
            mode = query.data.split('_')[1]
            self.user_search_mode[user_id] = mode
            mode_display = self._get_current_mode_display(user_id)
            try:
                keyboard = [
                    [
                        InlineKeyboardButton("По словам" + (" ✅" if mode == "words" else ""),
                                             callback_data="setmode_words"),
                        InlineKeyboardButton("Фраза" + (" ✅" if mode == "phrase" else ""),
                                             callback_data="setmode_phrase"),
                        InlineKeyboardButton("По смыслу" + (" ✅" if mode == "semantic" else ""),
                                             callback_data="setmode_semantic")
                    ],
                    [InlineKeyboardButton("🏠 В начало", callback_data="gohome")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await query.edit_message_text(
                    text=f"Режим поиска изменен на: <b>{mode_display}</b>.\nОтправьте ваш запрос.",
                    reply_markup=reply_markup,
                    parse_mode=ParseMode.HTML
                )
            except Exception as e:
                logger.warning(f"Не удалось изменить текст сообщения для setmode: {e}")
                await context.bot.send_message(chat_id=user_id,
                                               text=f"Режим поиска изменен на: {mode_display}.\nОтправьте ваш запрос.")
        else:
            logger.warning(f"Неизвестный callback_data: {query.data}")

    async def indexed_channels_command(self, update: Update, context: CallbackContext):
        if not self.channels_to_index_set:
            await update.message.reply_text("Список каналов для индексации не задан в конфигурации.")
            return
        response_lines = ["Я ищу по следующим каналам (если я в них состою):"]
        sorted_channels = sorted(list(self.channels_to_index_set))
        for ch_username in sorted_channels:
            response_lines.append(f"- @{ch_username}")
        response_text = "\n".join(response_lines)
        await update.message.reply_text(response_text)

    async def handle_search_query(self, update: Update, context: CallbackContext):
        if not update.message or not update.message.text:
            logger.info("Получено обновление без текстового сообщения, пропускаю.")
            return

        query_text_full = update.message.text.strip()
        user_id = update.effective_user.id

        if not query_text_full:
            await update.message.reply_text("Пожалуйста, введите непустой поисковый запрос.")
            return

        search_mode = self.user_search_mode.get(user_id, "words")
        mode_display = self._get_current_mode_display(user_id)

        await update.message.reply_text(f"Ищу \"{query_text_full}\" (режим: {mode_display})...")

        query_terms = []
        if search_mode == "words":
            query_terms = [term.lower() for term in re.findall(r'\b\w+\b', query_text_full)]
            if not query_terms:
                await update.message.reply_text("В режиме 'по словам' ваш запрос не содержит слов для поиска.")
                return

        logger.info(
            f"Search query: '{query_text_full}', mode: {search_mode}, parsed terms: {query_terms if query_terms else 'N/A'}")

        query_embedding = None
        if similarity_model and (
                search_mode == "semantic" or search_mode == "words"):
            query_embedding = similarity_model.encode(query_text_full)

        found_messages = self.db.search_messages(
            query_text=query_text_full,
            query_terms=query_terms,
            search_mode=search_mode,
            similarity_query_embedding=query_embedding,
            limit=5
        )

        home_button_keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("🏠 В начало", callback_data="gohome")]])

        if not found_messages:
            await update.message.reply_text("К сожалению, ничего не найдено по вашему запросу.",
                                            reply_markup=home_button_keyboard)
            return

        response_parts = []
        for i, msg_data in enumerate(found_messages):
            link = f"https://t.me/{msg_data['channel_username']}/{msg_data['telegram_message_id']}"
            text_preview = msg_data['text']
            match_type_display = msg_data.get("match_type", "неизвестно")

            if search_mode == "phrase" and match_type_display == "phrase" and query_text_full.lower() in text_preview.lower():
                start_index = text_preview.lower().find(query_text_full.lower())
                end_index = start_index + len(query_text_full)
                context_before = max(0, start_index - 50)
                context_after = min(len(text_preview), end_index + 50)
                highlighted_text = (text_preview[context_before:start_index].replace("<", "<").replace(">", ">") +
                                    f"<b>{text_preview[start_index:end_index].replace('<', '<').replace('>', '>')}</b>" +
                                    text_preview[end_index:context_after].replace("<", "<").replace(">", ">"))
                text_preview = ("..." if context_before > 0 else "") + highlighted_text + \
                               ("..." if context_after < len(text_preview) else "")
            else:
                text_preview = msg_data['text'][:200].replace("<", "<").replace(">", ">") + \
                               ("..." if len(msg_data['text']) > 200 else "")

            score_info = ""
            if msg_data['score'] == 1.0:
                score_info = f"(тип: {match_type_display})"
            elif "score" in msg_data:
                score_info = f"(схожесть: {msg_data['score']:.2f}, тип: {match_type_display})"
            date_str = msg_data['message_date'].strftime('%Y-%m-%d %H:%M') if msg_data['message_date'] else "N/A"
            response_parts.append(
                f"<b>Результат {i + 1}</b> {score_info}\n"
                f"📣 Канал: @{msg_data['channel_username']}\n"
                f"📝 {text_preview}\n"
                f"🔗 <a href='{link}'>Перейти к сообщению</a>\n"
                f"🗓️ {date_str}"
            )
        full_response = "\n\n---\n\n".join(response_parts)
        try:
            if len(full_response) <= Config.MAX_MESSAGE_LENGTH:
                await update.message.reply_html(full_response, disable_web_page_preview=True,
                                                reply_markup=home_button_keyboard)
            else:
                await update.message.reply_text(
                    "Найдено несколько сообщений. Отправляю по частям (форматирование может быть упрощено):",
                    reply_markup=home_button_keyboard
                )
                for part_html in response_parts:
                    try:
                        await update.message.reply_html(part_html, disable_web_page_preview=True,
                                                        reply_markup=home_button_keyboard)
                    except Exception as e_part:
                        logger.error(f"Ошибка отправки части результата: {e_part}")
                        await update.message.reply_text("Не удалось отправить часть результатов.",
                                                        reply_markup=home_button_keyboard)
                        break
        except Exception as e_send:
            logger.error(f"Ошибка отправки результатов поиска: {e_send}")
            await update.message.reply_text("Произошла ошибка при отправке результатов.",
                                            reply_markup=home_button_keyboard)

    def run(self):
        """Запускает бота (блокирующий вызов)."""
        logger.info("Запуск PTB поллинга...")
        self.ptb_application.run_polling()
        logger.info("PTB поллинг остановлен.")

if __name__ == '__main__':
    if not Config.BOT_TOKEN or \
            not Config.API_ID or \
            not Config.API_HASH:
        logger.error("Пожалуйста, установите корректные значения для BOT_TOKEN, API_ID и API_HASH в классе Config.")
    else:
        bot = NewsAggregatorBot()
        try:
            bot.run()
        except KeyboardInterrupt:
            logger.info("Принудительная остановка бота (Ctrl+C).")
        except Exception as e:
            logger.critical(f"Критическая ошибка в основном цикле: {e}", exc_info=True)
        finally:
            logger.info("Бот завершает работу.")
