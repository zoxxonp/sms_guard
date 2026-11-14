"""Админка: тексты SMS пользователей не показываются (ФТ-1.9), только метаданные проверок."""
from django.contrib import admin

from .models import Check, ModelVersion


@admin.register(Check)
class CheckAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "check_type", "category", "risk_level", "risk_score", "created_at")
    list_filter = ("check_type", "category", "risk_level")
    fields = ("user", "check_type", "language", "category", "risk_level", "risk_score", "p_sms", "p_url_max",
              "sms_model_version", "url_model_version", "processing_ms", "created_at")
    readonly_fields = fields

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False


@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    list_display = ("model_type", "version", "algorithm", "threshold", "trained_at", "is_active")
