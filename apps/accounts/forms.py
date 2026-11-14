"""Формы страниц входа, регистрации и профиля."""
from django import forms
from django.contrib.auth import authenticate, get_user_model
from django.contrib.auth.forms import PasswordChangeForm
from django.core.exceptions import ValidationError

from . import security


class LoginForm(forms.Form):
    username = forms.CharField(label="Логин", max_length=150,
                               widget=forms.TextInput(attrs={"autocomplete": "username", "autofocus": True}))
    password = forms.CharField(label="Пароль", strip=False,
                               widget=forms.PasswordInput(attrs={"autocomplete": "current-password"}))

    def __init__(self, request=None, *args, **kwargs):
        self.request = request
        self.user = None
        super().__init__(*args, **kwargs)

    def clean(self):
        username = self.cleaned_data.get("username", "")
        password = self.cleaned_data.get("password", "")
        if not username or not password:
            return self.cleaned_data
        remaining = security.lockout_remaining(username)
        if remaining:
            raise ValidationError(security.lockout_message(remaining), code="locked")
        user = authenticate(self.request, username=username, password=password)
        if user is None:
            security.register_failure(username)
            remaining = security.lockout_remaining(username)
            if remaining:
                raise ValidationError(security.lockout_message(remaining), code="locked")
            raise ValidationError("Неверный логин или пароль.", code="invalid_login")
        security.reset_failures(username)
        self.user = user
        return self.cleaned_data


class RegisterForm(forms.Form):
    username = forms.CharField(label="Логин", max_length=30,
                               help_text="3–30 символов: латинские буквы, цифры, «_»",
                               widget=forms.TextInput(attrs={"autocomplete": "username"}))
    email = forms.EmailField(label="E-mail", widget=forms.EmailInput(attrs={"autocomplete": "email"}))
    password1 = forms.CharField(label="Пароль", strip=False, help_text="Не менее 8 символов, буквы и цифры",
                                widget=forms.PasswordInput(attrs={"autocomplete": "new-password"}))
    password2 = forms.CharField(label="Повторите пароль", strip=False,
                                widget=forms.PasswordInput(attrs={"autocomplete": "new-password"}))

    def clean_username(self):
        return security.validate_username(self.cleaned_data["username"])

    def clean_email(self):
        return security.validate_email_unique(self.cleaned_data["email"])

    def clean(self):
        data = super().clean()
        p1, p2 = data.get("password1"), data.get("password2")
        if p1 and p2 and p1 != p2:
            self.add_error("password2", "Пароли не совпадают.")
        elif p1:
            try:
                security.validate_new_password(p1, get_user_model()(username=data.get("username", ""),
                                                                   email=data.get("email", "")))
            except ValidationError as exc:
                self.add_error("password1", exc)
        return data

    def save(self):
        return get_user_model().objects.create_user(username=self.cleaned_data["username"],
                                                    email=self.cleaned_data["email"],
                                                    password=self.cleaned_data["password1"])


class ConsentForm(forms.Form):
    store_sms_text = forms.BooleanField(
        label="Сохранять полный текст моих SMS в истории", required=False,
        help_text="По умолчанию выключено: хранится маскированный текст (номера карт, телефонов и суммы заменены на ***).")


class GuardPasswordChangeForm(PasswordChangeForm):
    def clean_new_password1(self):
        password = self.cleaned_data.get("new_password1")
        security.validate_new_password(password, self.user)
        return password


class DeleteHistoryForm(forms.Form):
    confirm = forms.BooleanField(label="Я понимаю, что все мои проверки будут удалены безвозвратно", required=True,
                                 error_messages={"required": "Подтвердите удаление, отметив флажок."})
