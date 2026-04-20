from recommender import ArtNailRecommender
import os
import scipy.sparse as sp
from tabulate import tabulate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sparse_path = os.path.join(BASE_DIR, '..', 'results', 'matrices', 'artnail_user_item_sparse_train.npz')
user_item_matrix = sp.load_npz(sparse_path)

# Настраиваем пути относительно папки src
art_nail = ArtNailRecommender(
    cb_model_path=os.path.join(BASE_DIR, '..', 'results', 'models', 'catboost_model.cbm'),
    ials_model_path=os.path.join(BASE_DIR, '..', 'results', 'models', 'best_IALS_model.pkl'),
    
    user_features_path=os.path.join(BASE_DIR, '..', 'results', 'feature_tables', 'user_features.pkl'),
    item_features_path=os.path.join(BASE_DIR, '..', 'results', 'feature_tables', 'item_features.pkl'),

    user_item_matrix=user_item_matrix,
    mappers_path=os.path.join(BASE_DIR, '..', 'results', 'mappers', 'id_mappers.pkl')
)


try:
    id_user = int(input("Enter user ID: "))
except ValueError:
    print("Invalid user ID. Please enter a valid integer.")
    exit()

def get_recommendations(user_id):
    return art_nail.recommend(user_id=user_id, top_n=5, category_cap=2)

def print_pretty_recs(df, user_id):
    print(f"\n{'='*50}")
    print(f" ПЕРСОНАЛЬНЫЕ РЕКОМЕНДАЦИИ ДЛЯ КЛИЕНТА {user_id} ")
    print(f"{'='*50}")

    display_df = df.copy()
    display_df['cb_score'] = (display_df['cb_score'] * 100).map("{:.2f}%".format)

    display_df.columns = ["Услуга", "Категория", "Вероятность"]

    print(tabulate(display_df, headers='keys', tablefmt='pretty', showindex=False))
    print(f"{'='*50}\n")


if __name__ == "__main__":
    recs = get_recommendations(id_user)
    if not recs.empty:
        print_pretty_recs(recs, id_user)
    else:
        print(f"К сожалению, не удалось сформировать рекомендации для ID_user = {id_user}. Возможно, это новый клиент без истории покупок.")