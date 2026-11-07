"""
Настройки Django для проекта SMS Guard.

Все секреты и параметры окружения читаются из файла .env (см. .env.example)
через django-environ. В репозитории хранится только .env.example (НФТ-7).
"""
from pathlib import Path

import environ

BASE_DIR = Path(__file__).resolve().parent.parent

env = environ.Env(
    DEBUG=(bool, False),
    ALLOWED_HOSTS=(list, ["127.0.0.1", "localhost"]),
    CSRF_TRUSTED_ORIGINS=(list, []),
    SESSION_COOKIE_SECURE=(bool, False),
    CSRF_COOKIE_SECURE=(bool, False),
    LOG_LEVEL=(str, "INFO"),
)
environ.Env.read_env(BASE_DIR / ".env")

# --- Базовые параметры -------------------------------------------------------
SECRET_KEY = env("SECRET_KEY")
DEBUG = env("DEBUG")
ALLOWED_HOSTS = env("ALLOWED_HOSTS")
CSRF_TRUSTED_ORIGINS = env("CSRF_TRUSTED_ORIGINS")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "apps.accounts",
    "apps.analysis",
    "apps.dashboard",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

# --- База данных -------------------------------------------------------------
# SQLite для разработки; для демонстрации задать в .env, например:
# DATABASE_URL=postgres://sms_guard:пароль@localhost:5432/sms_guard
DATABASE_URL = env.str("DATABASE_URL", default="") or f"sqlite:///{BASE_DIR / 'db.sqlite3'}"
DATABASES = {
    "default": env.db_url_config(DATABASE_URL),
}

# --- Пароли (ФТ-1.4: хеш с солью, PBKDF2 по умолчанию) -----------------------
PASSWORD_HASHERS = [
    "django.contrib.auth.hashers.PBKDF2PasswordHasher",
]
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
     "OPTIONS": {"min_length": 8}},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LOGIN_URL = "/login/"

# --- Cookie и сессии (ФТ-1.8) ------------------------------------------------
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"
SESSION_COOKIE_SECURE = env("SESSION_COOKIE_SECURE")  # True при работе по HTTPS
CSRF_COOKIE_SAMESITE = "Lax"
CSRF_COOKIE_SECURE = env("CSRF_COOKIE_SECURE")
X_FRAME_OPTIONS = "DENY"

# --- Локализация --------------------------------------------------------------
LANGUAGE_CODE = "ru"
TIME_ZONE = "Asia/Almaty"
USE_I18N = True
USE_TZ = True

# --- Статика ------------------------------------------------------------------
STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# --- Django REST Framework ------------------------------------------------------
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.SessionAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",
    ],
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ],
}

# --- Пути к ML-артефактам -----------------------------------------------------
ARTIFACTS_DIR = BASE_DIR / "ml" / "artifacts"
RISK_CONFIG_PATH = BASE_DIR / "risk_config.yaml"

# --- Логирование (НФТ-13): logs/app.log, ротация 5 файлов по 1 МБ --------------
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            # уровень, время, модуль, сообщение
            "format": "%(levelname)s %(asctime)s %(name)s %(module)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": LOG_DIR / "app.log",
            "maxBytes": 1024 * 1024,  # 1 МБ
            "backupCount": 4,         # app.log + app.log.1..4 = 5 файлов
            "encoding": "utf-8",
            "formatter": "standard",
        },
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
    },
    "root": {
        "handlers": ["file", "console"],
        "level": env("LOG_LEVEL"),
    },
    "loggers": {
        "django": {"handlers": ["file", "console"], "level": "INFO", "propagate": False},
        "sms_guard": {"handlers": ["file", "console"], "level": env("LOG_LEVEL"), "propagate": False},
    },
}
