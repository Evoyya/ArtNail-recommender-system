import asyncio
import os
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
async def cdm_start(messege: types.Message):
    await messege.answer("👋 Привет! Я помощник ArtNail.\nПришли мне **ID клиента**, чтобы получить персональные рекомендации.")

@dp.message()
async def handle_id(messege: types.Message):
    if messege.text.isdigit():
        user_id = int(messege.text)
        await messege.answer(f"🔎 Ищу лучшие предложения для клиента {user_id}...")
        try:
            recs = recommender.recommend(user_id=user_id, top_n=5, category_cap=2)

            if recs.empty:
                await messege.answer("❌ Не удалось найти рекомендации для этого ID.")
                return 
        
            response = f"✨ **Топ-услуги для клиента {user_id}:**\n\n"
            for i, row in recs.iterrows():
                response += f"🔥 **{row['item_name']}**\n"
                response += f"📂 Категория: {row['item_category']}\n"
                response += f"📈 Вероятность: {row['cb_score']:.1%}\n"
                response += "----------------------------\n"
            
            await messege.answer(response, parse_mode="Markdown")
        except Exception as e:
            await messege.answer(f"⚠️ Произошла ошибка при получении рекомендаций: {str(e)}")
    else:
        await messege.answer("❌ Пожалуйста, отправьте корректный числовой ID клиента.")

async def main():
    print("Бот запущен...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main()) 
            
