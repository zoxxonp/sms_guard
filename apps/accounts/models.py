"""Профиль пользователя (раздел 4.3 ТЗ, таблица user_profile)."""
from django.conf import settings
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver


class UserProfile(models.Model):
    LANGUAGES = [("ru", "Русский"), ("kk", "Қазақша"), ("en", "English")]

    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="profile",
                                verbose_name="пользователь")
    store_sms_text = models.BooleanField("согласие на хранение полного текста SMS", default=False)
    preferred_language = models.CharField("язык интерфейса", max_length=2, choices=LANGUAGES, default="ru")

    class Meta:
        db_table = "user_profile"
        verbose_name = "профиль"
        verbose_name_plural = "профили"

    def __str__(self) -> str:
        return f"Профиль {self.user}"


@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def create_profile(sender, instance, created, **kwargs):
    """Профиль создаётся вместе с пользователем; согласие по умолчанию выключено."""
    if created:
        UserProfile.objects.get_or_create(user=instance)


def get_profile(user) -> UserProfile:
    profile, _ = UserProfile.objects.get_or_create(user=user)
    return profile
