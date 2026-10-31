"""API приложения analysis."""
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView


class HealthView(APIView):
    """GET /api/v1/health/ — проверка, что приложение запущено.

    Доступен без авторизации. В задании 12 сюда добавится статус загруженных моделей.
    """

    permission_classes = [AllowAny]
    authentication_classes: list = []

    def get(self, request):
        return Response({"status": "ok"})
