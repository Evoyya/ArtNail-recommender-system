import pandas as pd
import joblib
from catboost import CatBoostClassifier
from utils import IDMapper
import os




class ArtNailRecommender:
    """
        Класс-рекомендатель для проекта ArtNail, реализующий логику Two-Stage Ranking.

        Система сначала генерирует кандидатов с помощью модели коллаборативной 
        фильтрации (iALS), затем обогащает их признаками пользователей/услуг 
        и переранжирует с помощью градиентного бустинга (CatBoost).

        Attributes:
            cb_model: Загруженная модель CatBoostClassifier для финального ранжирования.
            ials_model: Загруженная модель implicit.als для генерации кандидатов.
            user_features (pd.DataFrame): Таблица с признаками пользователей.
            item_features (pd.DataFrame): Таблица с признаками и метаданными услуг.
            user_item_matrix (csr_matrix): Матрица взаимодействий для iALS.
            mappers (dict): Словарь с объектами IDMapper для пользователей и товаров.
            features_list (list): Список имен признаков, необходимых для CatBoost.
        """
    def __init__(self, cb_model_path, ials_model_path, user_features_path, item_features_path, user_item_matrix, mappers_path):
        """
        Инициализирует рекомендатель, загружая все необходимые артефакты.

        Args:
            cb_model_path (str): Путь к файлу модели CatBoost.
            ials_model_path (str): Путь к файлу модели iALS.
            user_features_path (str): Путь к дампу признаков пользователей.
            item_features_path (str): Путь к дампу признаков услуг.
            user_item_matrix: Матрица взаимодействий в формате CSR.
            mappers_path (str): Путь к дампу словарей маппинга (IDMapper).
        """
        # Загружаем всё при инициализации класса
        self.cb_model = CatBoostClassifier().load_model(cb_model_path)
        self.ials_model = joblib.load(ials_model_path)
        self.user_features = joblib.load(user_features_path)
        self.item_features = joblib.load(item_features_path)
        self.user_item_matrix = user_item_matrix
        self.mappers = joblib.load(mappers_path)
        
        # Список фичей, которые ожидает CatBoost
        self.features_list = self.cb_model.feature_names_

    def recommend(self, user_id, top_n=5, category_cap=2):
        """
        Формирует персональные рекомендации для пользователя.

        Процесс включает:
        1. Проверку пользователя на "холодный старт".
        2. Отбор топ-50 кандидатов через iALS.
        3. Feature Engineering (слияние кандидатов с признаками).
        4. Ранжирование кандидатов через CatBoost.
        5. Пост-фильтрацию для обеспечения разнообразия (diversity) категорий.

        Args:
            user_id (int/str): Внешний идентификатор пользователя.
            top_n (int): Количество итоговых рекомендаций. По умолчанию 5.
            category_cap (int): Максимальное количество услуг одной категории 
                в выдаче. По умолчанию 2.

        Returns:
            pd.DataFrame: Таблица с колонками ['item_name', 'item_category', 'cb_score'],
                отсортированная по убыванию релевантности.
        """
        user_mapper = self.mappers['user_mapper']
        item_mapper = self.mappers['item_mapper']
        user_idx = user_mapper.id_to_idx.get(user_id)

        if user_idx is None:
            # Обработка холодного старта, если пользователя нет в матрице
            return self.item_features.sort_values('item_unique_users', ascending=False).head(top_n)
        
        # 1. Генерация кандидатов через iALS
        ids, scores = self.ials_model.recommend(user_idx, self.user_item_matrix[user_idx], N=50)
        real_item_ids = [item_mapper.idx_to_id[idx] for idx in ids]

        user_cands = pd.DataFrame({
            'id_item': real_item_ids,
            'ials_score': scores
        })
        user_cands['id_user'] = user_id

        # 2. Приклеиваем статику услуг (имя, категория, популярность)
        user_cands = user_cands.merge(self.item_features, on='id_item', how='left')

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
        """
        Ограничивает количество повторений одной категории в финальной выдаче.

        Args:
            df (pd.DataFrame): Кандидаты со скорами.
            cap (int): Лимит элементов на одну категорию.
            top_n (int): Общее количество необходимых рекомендаций.

        Returns:
            pd.DataFrame: Отфильтрованные и отсортированные рекомендации.
        """
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