"""API приложения analysis."""
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from ml.predictor import get_predictor


class HealthView(APIView):
    """GET /api/v1/health/ — статус приложения и загруженных моделей. Доступен без авторизации."""

    permission_classes = [AllowAny]
    authentication_classes: list = []

    def get(self, request):
        # Текст ошибки (пути к файлам) наружу не отдаётся — он есть в logs/app.log
        status = {kind: {k: v for k, v in info.items() if k != "error"}
                  for kind, info in get_predictor().status().items()}
        ok = status["sms"]["loaded"] and status["url"]["loaded"]
        return Response({"status": "ok" if ok else "degraded", "models": status})
