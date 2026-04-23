import asyncio
import os
import pandas as pd
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from dotenv import load_dotenv
from aiogram import F
import scipy.sparse as sp
from recommender import ArtNailRecommender
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Начинаю загрузку моделей... Пожалуйста, подожди.")


load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sparse_path = os.path.join(BASE_DIR, '..', 'results', 'matrices', 'artnail_user_item_sparse_train.npz')
user_item_matrix = sp.load_npz(sparse_path)

users_clean_path = os.path.join(BASE_DIR, '..', 'Tables', 'CleanTable', 'users_clean.csv') 
users_df = pd.read_csv(users_clean_path)

users_df['Телефон'] = users_df['Телефон'].astype(str).str.replace(r'\D', '', regex=True)
phone_to_id = dict(zip(users_df['Телефон'], users_df['id_user']))

#Объект рекомендательной системы
recommender = ArtNailRecommender(
    cb_model_path=os.path.join(BASE_DIR, '..', 'results', 'models', 'catboost_model.cbm'),
    ials_model_path=os.path.join(BASE_DIR, '..', 'results', 'models', 'best_IALS_model.pkl'),
    user_features_path=os.path.join(BASE_DIR, '..', 'results', 'feature_tables', 'user_features.pkl'),
    item_features_path=os.path.join(BASE_DIR, '..', 'results', 'feature_tables', 'item_features.pkl'),
    user_item_matrix=user_item_matrix,
    mappers_path=os.path.join(BASE_DIR, '..', 'results', 'mappers', 'id_mappers.pkl')
)


#Настройки бота
BOT_TOKEN = os.getenv("BOT_TOKEN")
bot = Bot(token=BOT_TOKEN)  
dp = Dispatcher()

@dp.message(Command("start"))
async def cdm_start(message: types.Message):
    await message.answer("👋 Привет! Я помощник ArtNail.\nПришли мне номер телефона или ID клиента, чтобы получить персональные рекомендации.")

@dp.message()
async def handle_message(message: types.Message):
    # Очистка ввода: оставляем только цифры
    raw_text = "".join(filter(str.isdigit, message.text))
    
    user_id = None

    if not raw_text:
        await message.answer("❌ Пожалуйста, введите ID клиента или номер телефона.")
        return

    # Логика определения: телефон или ID
    if len(raw_text) >= 10:  
        
        user_id = phone_to_id.get(raw_text)
        
        # Если не нашли, пробуем проверить вариант с '7' вместо '8' в начале
        if not user_id:
            if raw_text.startswith('8'):
                user_id = phone_to_id.get('7' + raw_text[1:])
            elif raw_text.startswith('7'):
                user_id = phone_to_id.get('8' + raw_text[1:])

        if not user_id:
            await message.answer(f"📱 Клиент с номером `{raw_text}` не найден в базе.")
            return
    else:
        # Считаем, что это прямой ID
        user_id = int(raw_text)

    # процесс рекомендации
    await message.answer(f"🔎 Ищу рекомендации для клиента (ID: {user_id})...")
    try:
        recs = recommender.recommend(user_id=user_id, top_n=5, category_cap=2)

        if recs.empty:
            await message.answer("🤷‍♂️ Не удалось найти рекомендации для этого клиента.")
            return 
    
        response = f"✨ **Рекомендации для клиента {user_id}:**\n\n"
        for i, row in recs.iterrows():
            response += f"✅ **{row['item_name']}**\n"
            response += f"🔹 _{row['item_category']}_\n"
            response += f"📊 Вероятность: {row['cb_score']:.1%}\n"
            response += "⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯\n"
        
        await message.answer(response, parse_mode="Markdown")
        
    except Exception as e:
        await message.answer(f"⚠️ Ошибка: {str(e)}")

async def main():
    print("Бот запущен...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main()) 
            
