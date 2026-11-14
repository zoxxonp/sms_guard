import os
import sys

from django.apps import AppConfig


class AnalysisConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.analysis"
    label = "analysis"
    verbose_name = "Анализ SMS и URL"

    def ready(self):
        """Модели загружаются один раз при старте сервера (а не при первом запросе).

        Для runserver — только в рабочем процессе автоперезагрузчика (RUN_MAIN) или при --noreload;
        для WSGI-сервера загрузка выполняется в config/wsgi.py. Команды migrate, test и т. п. модели не грузят.
        """
        if len(sys.argv) > 1 and sys.argv[1] == "runserver" and (
                os.environ.get("RUN_MAIN") == "true" or "--noreload" in sys.argv):
            from ml.predictor import get_predictor
            get_predictor()
