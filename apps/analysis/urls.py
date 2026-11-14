"""Страницы проверки, результата и истории (раздел 8.1 ТЗ)."""
from django.urls import path

from . import views

app_name = "analysis"

urlpatterns = [
    path("check/sms/", views.check_sms_page, name="check-sms"),
    path("check/url/", views.check_url_page, name="check-url"),
    path("checks/<int:pk>/", views.check_detail_page, name="check-detail"),
    path("checks/<int:pk>/delete/", views.check_delete_page, name="check-delete"),
    path("history/", views.history_page, name="history"),
]
