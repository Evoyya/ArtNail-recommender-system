import pandas as pd
import joblib
from catboost import CatBoostClassifier

class ArtNailRecommender:
    def __init__(self, cb_model_path, ials_model_path, user_features_path, item_features_path):
        # Загружаем всё при инициализации класса
        self.cb_model = CatBoostClassifier().load_model(cb_model_path)
        self.ials_model = joblib.load(ials_model_path)
        self.user_features = joblib.load(user_features_path)
        self.item_features = joblib.load(item_features_path)

        # Список фичей, которые ожидает CatBoost
        self.features_list = self.cb_model.feature_names_

    def recommend(self, user_id, top_n=5, category_cap=2):
        # 1. Генерация кандидатов через iALS
        # iALS возвращает индексы и скоры. Мы берем [0], так как юзер один
        ids, scores = self.ials_model.recommend([user_id], n=50)
        
        user_cands = pd.DataFrame({
            'id_item': ids[0],
            'ials_score': scores[0]
        })
        user_cands['id_user'] = user_id

        # 2. Приклеиваем статику услуг (имя, категория, популярность)
        user_cands = user_cands.merge(self.item_stats, on='id_item', how='left')

        # 3. Приклеиваем фичи юзера
        u_feat = self.user_features[self.user_features['id_user'] == user_id]
        
        if not u_feat.empty:
            # Если юзер есть в базе, копируем его показатели
            for col in ['user_total_spent', 'avg_visit_cycle']:
                user_cands[col] = u_feat[col].values[0]
        else:
            # Если юзер новый, ставим заглушки
            user_cands['user_total_spent'] = 0.0
            user_cands['avg_visit_cycle'] = 365.0

        # Заполняем NaN, если они просочились
        user_cands['avg_visit_cycle'] = user_cands['avg_visit_cycle'].fillna(365.0)
        user_cands['user_total_spent'] = user_cands['user_total_spent'].fillna(0.0)

        user_cands['cb_score'] = self.cb_model.predict_proba(user_cands[self.features_list])[:, 1]

        # 5. Применяем лимит на категории (Diversity)
        return self._apply_category_cap(user_cands, cap=category_cap, top_n=top_n)
    
    def _apply_category_cap(self, df, cap, top_n):
        # Сортируем по итоговому скору бустинга
        sorted_df = df.sort_values('cb_score', ascending=False)
        
        final_recs = []
        cat_counts = {}

        for _, row in sorted_df.iterrows():
            cat = row['item_category']
            count = cat_counts.get(cat, 0)
            
            if count < cap:
                final_recs.append(row)
                cat_counts[cat] = count + 1
            
            if len(final_recs) >= top_n:
                break
                
        result = pd.DataFrame(final_recs)
        return result[['item_name', 'item_category', 'cb_score']]