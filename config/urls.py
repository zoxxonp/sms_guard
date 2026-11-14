"""Корневая маршрутизация SMS Guard."""
from django.contrib import admin
from django.templatetags.static import static
from django.urls import include, path
from django.views.generic import RedirectView

urlpatterns = [
    path("favicon.ico", RedirectView.as_view(url=static("img/favicon.svg"), permanent=True)),
    path("admin/", admin.site.urls),
    path("api/v1/", include("apps.analysis.api_urls")),
    path("", include("apps.accounts.urls")),
    path("", include("apps.analysis.urls")),
    path("", include("apps.dashboard.urls")),
]
