"""WSGI-точка входа SMS Guard."""
import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

application = get_wsgi_application()

# Загрузка моделей при старте процесса (раздел 2.4 ТЗ): сообщение в logs/app.log сразу при запуске
from ml.predictor import get_predictor  # noqa: E402

get_predictor()
