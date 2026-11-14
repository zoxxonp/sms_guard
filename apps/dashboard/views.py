"""Dashboard (/) и расширенная статистика (/stats/). Данные графиков — из /api/v1/stats/summary/."""
from django.contrib.auth.decorators import login_required
from django.shortcuts import render

from apps.analysis import stats


@login_required
def home(request):
    days = stats.parse_period(request.GET.get("days"))
    return render(request, "dashboard/home.html", {"days": days, "periods": stats.PERIODS, "active": "home",
                                                   "is_admin": request.user.is_staff})


@login_required
def stats_page(request):
    days = stats.parse_period(request.GET.get("days"))
    return render(request, "dashboard/stats.html", {"days": days, "periods": stats.PERIODS, "active": "stats",
                                                    "is_admin": request.user.is_staff})
