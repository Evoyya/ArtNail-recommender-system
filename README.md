# Recommendation System for Beauty Salon
Рекомендательная система полного цикла, разработанная для повышения среднего чека и лояльности клиентов в салоне красоты. Проект включает в себя сбор данных, построение гибридной модели рекомендаций и интерфейс в виде Telegram-бота для администраторов.

## Key ML Features
- Hybrid Approach: Сочетание матричной факторизации (iALS) и градиентного бустинга (CatBoost).

- Two-Stage Ranking: Отбор кандидатов через iALS и последующее ранжирование с использованием признаков пользователей и услуг.

- Cold Start Handling: Стратегия работы с новыми пользователями через популярные услуги и контентные признаки.

## ML Pipeline Architecture
Проект реализован в виде классического пайплайна:

1) Data Preprocessing & EDA:

    * Очистка данных из ERP-системы (YCLIENTS).

    * Анализ распределения покупок и жизненного цикла клиента (LTV).

    * Обработка пропусков и генерация признаков (Feature Engineering).

2) Candidate Generation (iALS):

    * Использование библиотеки implicit для построения матрицы взаимодействий.

    * Обучение эмбеддингов пользователей и айтемов.

    * HitRate@50 используется как основная метрика на данном этапе.

3) Ranking (CatBoost):

    * Формирование обучающей выборки с использованием позитивных и негативных примеров.

    * Генерация признаков: частота посещений, средний чек, категории-фавориты, временные лаги.

    * Оптимизация модели для предсказания вероятности покупки конкретной услуги.

4) Evaluation:

    * Валидация на отложенной выборке (Time-based split).

    * Метрики: HitRate@10, Recall@k, MRR.


## Tech Stack
Languages: Python (3.10+)

Data Science: Pandas, NumPy, Scikit-learn, SciPy

ML Models: CatBoost, Implicit (iALS)

Optimization: Optuna (Hyperparameter tuning)

Deployment: Aiogram (Telegram Bot), Docker, python-dotenv

## 📁 Project Structure

```text
ARTNAIL/
├── results/                  # Артефакты обучения и предобработки
│   ├── feature_tables/       # Сгенерированные признаки (user/item features)
│   ├── mappers/              # Маппинги внутренних ID в реальные идентификаторы
│   ├── matrices/             # Разреженные матрицы (sparse) для обучения iALS
│   └── models/               # Сохраненные веса (CatBoost .cbm, iALS .pkl)
├── src/                      # Исходный код системы
│   ├── Notebooks/            # Исследовательская часть (Jupyter Notebooks)
│   │   ├── DataDesing/       # Очистка и инженерия признаков
│   │   └── Models/           # Эксперименты с CatBoost и iALS
│   ├── bot.py                # Реализация Telegram-интерфейса (aiogram)
│   ├── recommender.py        # Ядро системы (класс ArtNailRecommender)
│   ├── utils.py              # Вспомогательные утилиты и ID-мапперы
│   └── main.py               # Точка входа (опционально)
├── Tables/                   # База данных проекта
│   ├── BaseTable/            # Сырые данные из YCLIENTS
│   └── CleanTable/           # Очищенные данные для обучения
├── Dockerfile                # Конфигурация для контейнеризации
├── .dockerignore             # Исключения для сборки образа
├── .gitignore                # Исключения для Git 
├── requirements.txt          # Минимальные зависимости для Production
└── requirements-dev.txt      # Полный набор для разработки и DS-анализа
