from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config


NEGATIVE_INTENTS = [
    "хочу закрыть аккаунт",
    "планирую отказаться от подписки",
    "сервис стал слишком дорогим",
    "постоянные ошибки и зависания",
    "нашел более выгодную альтернативу",
    "поддержка не решила мой вопрос",
    "не вижу пользы от продления",
    "качество связи сильно упало",
    "прошу отменить автопродление",
    "верните деньги за последний месяц",
]

POSITIVE_INTENTS = [
    "хочу продлить подписку",
    "все работает стабильно",
    "подскажите как подключить новый тариф",
    "сервис нравится нашей команде",
    "нужно добавить еще пользователей",
    "оплата прошла успешно",
    "хочу включить дополнительные функции",
    "спасибо за быстрое решение вопроса",
    "планируем пользоваться дальше",
    "нужна консультация по настройкам",
]

NEUTRAL_CONTEXT = [
    "после последнего обновления",
    "в мобильном приложении",
    "в личном кабинете",
    "для корпоративного аккаунта",
    "при оплате картой",
    "в вечернее время",
    "на домашнем интернете",
    "у нескольких сотрудников",
    "после смены тарифа",
    "в течение этой недели",
]

CHANNELS = np.array(["chat", "email", "phone", "web_form"])
SEGMENTS = np.array(["individual", "small_business", "enterprise"])


def generate_churn_text_data(n_rows: int = 100_000, random_state: int = 42) -> pd.DataFrame:
    """Generate a synthetic text classification dataset for churn intent."""
    rng = np.random.default_rng(random_state)
    churn = rng.binomial(1, 0.42, size=n_rows)
    urgency = rng.integers(1, 6, size=n_rows)
    months_active = rng.integers(1, 73, size=n_rows)
    support_tickets = rng.poisson(np.where(churn == 1, 2.4, 0.8)).clip(0, 12)
    discount_requested = rng.binomial(1, np.where(churn == 1, 0.48, 0.18))
    channel = rng.choice(CHANNELS, size=n_rows, p=[0.42, 0.31, 0.17, 0.10])
    segment = rng.choice(SEGMENTS, size=n_rows, p=[0.55, 0.32, 0.13])

    negative = rng.choice(NEGATIVE_INTENTS, size=n_rows)
    positive = rng.choice(POSITIVE_INTENTS, size=n_rows)
    context = rng.choice(NEUTRAL_CONTEXT, size=n_rows)
    secondary = rng.choice(NEGATIVE_INTENTS + POSITIVE_INTENTS, size=n_rows)

    texts = np.where(
        churn == 1,
        [
            f"{main}, {ctx}. Срочность {urg}/5. Обращений в поддержку: {tickets}. "
            f"{'Нужна скидка, иначе уйду.' if discount else 'Прошу решить вопрос сегодня.'}"
            for main, ctx, urg, tickets, discount in zip(
                negative, context, urgency, support_tickets, discount_requested
            )
        ],
        [
            f"{main}, {ctx}. Срочность {urg}/5. Обращений в поддержку: {tickets}. "
            f"{'Интересует скидка на годовой тариф.' if discount else 'Хочу продолжить использование.'}"
            for main, ctx, urg, tickets, discount in zip(
                positive, context, urgency, support_tickets, discount_requested
            )
        ],
    )

    # Add a little noise so the task is not solved by one exact phrase.
    noisy_mask = rng.random(n_rows) < 0.12
    texts = np.where(noisy_mask, np.char.add(np.char.add(texts, " Дополнительно: "), secondary), texts)

    return pd.DataFrame(
        {
            "message": texts,
            "urgency": urgency,
            "months_active": months_active,
            "support_tickets": support_tickets,
            "discount_requested": discount_requested,
            "channel": channel,
            "segment": segment,
            "churn_intent": churn,
        }
    )


def save_dataset(path: Path, n_rows: int = 100_000, random_state: int = 42) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = generate_churn_text_data(n_rows=n_rows, random_state=random_state)
    data.to_csv(path, index=False)
    return data


def main() -> None:
    config = load_config()
    data = save_dataset(config.data_path, n_rows=config.dataset_rows, random_state=config.random_state)
    print(f"Saved {len(data)} rows to {config.data_path}")


if __name__ == "__main__":
    main()
