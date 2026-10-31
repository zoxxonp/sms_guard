"""Страницы приложения dashboard."""
from django.shortcuts import render


def home(request):
    """Главная страница-заглушка (задание 2). Dashboard появится в задании 17."""
    return render(request, "dashboard/home.html")
