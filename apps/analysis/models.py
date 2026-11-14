"""Модели хранения проверок (раздел 4.3 ТЗ)."""
from django.conf import settings
from django.db import models


class CheckType(models.TextChoices):
    SMS = "SMS", "SMS"
    URL = "URL", "Ссылка"


class Category(models.TextChoices):
    SAFE = "SAFE", "Безопасное"
    SUSPICIOUS = "SUSPICIOUS", "Подозрительное"
    FRAUDULENT = "FRAUDULENT", "Мошенническое"
    DANGEROUS = "DANGEROUS", "Опасная ссылка"


class RiskLevel(models.TextChoices):
    LOW = "LOW", "Низкий"
    MEDIUM = "MEDIUM", "Средний"
    HIGH = "HIGH", "Высокий"


# Для сводной статистики: опасная ссылка считается вместе с мошенническими сообщениями
FRAUD_CATEGORIES = (Category.FRAUDULENT, Category.DANGEROUS)


class Check(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="checks",
                             verbose_name="пользователь")
    check_type = models.CharField("тип", max_length=3, choices=CheckType.choices)
    input_text = models.TextField("полный текст (только с согласия)", null=True, blank=True)
    input_masked = models.TextField("маскированный текст")
    input_hash = models.CharField("HMAC ввода (защита от повторной отправки)", max_length=64, db_index=True)
    language = models.CharField("язык", max_length=5, null=True, blank=True)
    category = models.CharField("категория", max_length=10, choices=Category.choices)
    risk_level = models.CharField("уровень риска", max_length=6, choices=RiskLevel.choices)
    risk_score = models.PositiveSmallIntegerField("Risk Score")
    p_sms = models.FloatField("P_sms", null=True, blank=True)
    p_url_max = models.FloatField("P_url,max", null=True, blank=True)
    explanation = models.JSONField("причины", default=list)
    recommendation = models.TextField("рекомендация")
    language_note = models.TextField("предупреждение о языке", null=True, blank=True)
    sms_model_version = models.CharField("версия SMS-модели", max_length=20, null=True, blank=True)
    url_model_version = models.CharField("версия URL-модели", max_length=20, null=True, blank=True)
    processing_ms = models.FloatField("время анализа, мс", default=0)
    created_at = models.DateTimeField("дата проверки", auto_now_add=True, db_index=True)

    class Meta:
        db_table = "checks"
        ordering = ["-created_at", "-id"]
        indexes = [models.Index(fields=["user", "created_at"], name="checks_user_created_idx"),
                   models.Index(fields=["category"], name="checks_category_idx")]
        verbose_name = "проверка"
        verbose_name_plural = "проверки"

    def __str__(self) -> str:
        return f"#{self.pk} {self.check_type} {self.category} {self.risk_score}"

    @property
    def display_text(self) -> str:
        return self.input_text if self.input_text is not None else self.input_masked

    @property
    def fragment(self) -> str:
        text = " ".join(self.display_text.split())
        return text if len(text) <= 60 else text[:59] + "…"

    @property
    def is_fraud(self) -> bool:
        return self.category in FRAUD_CATEGORIES


class UrlResult(models.Model):
    check_ref = models.ForeignKey(Check, on_delete=models.CASCADE, related_name="url_results",
                                  db_column="check_id", verbose_name="проверка")
    url = models.CharField("ссылка", max_length=2048)
    domain = models.CharField("домен", max_length=255)
    p_url = models.FloatField("P_url")
    category = models.CharField("категория ссылки", max_length=10)
    features_json = models.JSONField("признаки и причины", default=dict)

    class Meta:
        db_table = "url_results"
        ordering = ["id"]
        verbose_name = "результат по ссылке"
        verbose_name_plural = "результаты по ссылкам"


class DetectedFeature(models.Model):
    check_ref = models.ForeignKey(Check, on_delete=models.CASCADE, related_name="detected_features",
                                  db_column="check_id", verbose_name="проверка")
    code = models.CharField("код", max_length=30, db_index=True)
    title = models.CharField("название", max_length=200)
    weight = models.FloatField("вклад", default=0)

    class Meta:
        db_table = "detected_features"
        ordering = ["id"]
        verbose_name = "обнаруженный признак"
        verbose_name_plural = "обнаруженные признаки"


class ModelVersion(models.Model):
    model_type = models.CharField("тип", max_length=3, choices=[("sms", "SMS"), ("url", "URL")])
    version = models.CharField("версия", max_length=20)
    algorithm = models.CharField("алгоритм", max_length=100)
    metrics_json = models.JSONField("метрики на test", default=dict)
    threshold = models.FloatField("порог", null=True, blank=True)
    trained_at = models.CharField("дата обучения", max_length=32, blank=True)
    is_active = models.BooleanField("активна", default=False)

    class Meta:
        db_table = "model_versions"
        unique_together = [("model_type", "version", "trained_at")]
        ordering = ["model_type", "-id"]
        verbose_name = "версия модели"
        verbose_name_plural = "версии моделей"

    def __str__(self) -> str:
        return f"{self.model_type} {self.version} ({self.algorithm})"
