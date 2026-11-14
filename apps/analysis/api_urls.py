"""Маршруты REST API v1 (раздел 4.4 ТЗ)."""
from django.urls import path

from apps.accounts import api as auth_api

from . import api, views

app_name = "api"

urlpatterns = [
    path("auth/register/", auth_api.RegisterView.as_view(), name="register"),
    path("auth/login/", auth_api.LoginView.as_view(), name="login"),
    path("auth/logout/", auth_api.LogoutView.as_view(), name="logout"),
    path("auth/csrf/", auth_api.CsrfView.as_view(), name="csrf"),
    path("check/sms/", api.CheckSmsView.as_view(), name="check-sms"),
    path("check/url/", api.CheckUrlView.as_view(), name="check-url"),
    path("checks/", api.CheckListView.as_view(), name="checks"),
    path("checks/<int:pk>/", api.CheckDetailView.as_view(), name="check-detail"),
    path("stats/summary/", api.StatsSummaryView.as_view(), name="stats-summary"),
    path("health/", views.HealthView.as_view(), name="health"),
]
