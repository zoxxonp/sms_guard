"""python manage.py seed_demo — демонстрационный пользователь и 30 проверок (задание 16, п. 8).

Все сообщения искусственные (собственный набор задания 3 и демонстрационные сообщения раздела 8.3),
ссылки — только зарезервированные домены. Результаты рассчитываются обученными моделями при запуске команды,
а не берутся из заготовки; даты проверок распределены по последним 30 дням.
"""
import random
from datetime import timedelta
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from apps.accounts.models import get_profile
from apps.analysis.models import Check
from apps.analysis.services import analyze_sms, analyze_url
from apps.analysis.storage import save_result
from ml.predictor import get_predictor

DEMO_MESSAGES = [
    "Привет! Встречаемся завтра в 18:00 у входа в библиотеку.",
    "Your order #4821 has been shipped and will arrive on Monday.",
    "Сәлем! Ертең сағат 10:00-де кездесеміз бе?",
    "Ваш код подтверждения: 4821. Никому не сообщайте этот код.",
    "Ваша карта заблокирована! Срочно подтвердите данные по ссылке: http://secure-bank-verify.example/login",
    "URGENT: Your account will be suspended in 24 hours. Verify now at http://192.0.2.15/account-update",
    "Мама, у меня проблемы, срочно переведи 50 000 на этот номер, потом всё объясню. Никому не говори.",
    "Congratulations! You won a $1000 gift card. Claim your prize: bit.ly/prize-claim-test",
    "Сотрудник службы безопасности банка: назовите код из SMS для отмены подозрительной операции.",
    "Картаңыз бұғатталды. Жедел түрде мына сілтеме арқылы растаңыз: http://bank-kz-verify.test",
    "Ваша посылка ожидает оплаты таможенного сбора 350 T: http://post-delivery-pay.example",
]
DEMO_URLS = [
    "https://www.example.com/about",
    "http://xn--exmple-cua.test/secure/login?verify=1",
    "https://login-account.example.com/update",
]
TOTAL = 30


class Command(BaseCommand):
    help = "Создаёт пользователя demo_user и 30 демонстрационных проверок"

    def add_arguments(self, parser):
        parser.add_argument("--username", default="demo_user")
        parser.add_argument("--password", default="DemoPass2026",
                            help="Пароль демо-пользователя (по умолчанию DemoPass2026 — только для демонстрации)")
        parser.add_argument("--keep", action="store_true", help="Не удалять прежние проверки демо-пользователя")
        parser.add_argument("--seed", type=int, default=42)

    def handle(self, *args, **opts):
        predictor = get_predictor()
        if not (predictor.sms_ready and predictor.url_ready):
            raise CommandError("Модели не загружены. Сначала: python -m ml.train_sms и python -m ml.train_url")

        User = get_user_model()
        user, created = User.objects.get_or_create(username=opts["username"],
                                                   defaults={"email": f"{opts['username']}@example.com"})
        user.set_password(opts["password"])
        user.save()
        profile = get_profile(user)
        profile.store_sms_text = True          # демонстрационные тексты искусственные — храним полностью
        profile.save(update_fields=["store_sms_text"])
        if not opts["keep"]:
            Check.objects.filter(user=user).delete()

        rng = random.Random(opts["seed"])
        custom = pd.read_csv(Path(settings.BASE_DIR) / "data" / "raw" / "custom_ru_kk.csv")
        extra_n = TOTAL - len(DEMO_MESSAGES) - len(DEMO_URLS)
        extra = custom.sample(n=extra_n, random_state=opts["seed"])["text"].tolist()

        items = [("SMS", t) for t in DEMO_MESSAGES + extra] + [("URL", u) for u in DEMO_URLS]
        rng.shuffle(items)
        now = timezone.now()
        counts = {}
        for i, (kind, value) in enumerate(items):
            result = analyze_sms(value, predictor) if kind == "SMS" else analyze_url(value, predictor)
            when = now - timedelta(days=rng.randint(0, 29), hours=rng.randint(0, 10), minutes=rng.randint(0, 59))
            check = save_result(user, result, predictor, created_at=when)
            counts[check.category] = counts.get(check.category, 0) + 1

        self.stdout.write(self.style.SUCCESS(
            f"{'Создан' if created else 'Обновлён'} пользователь {user.username} (пароль: {opts['password']}); "
            f"проверок: {len(items)} — " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))))
