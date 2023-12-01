"""
Django settings for evapp2 project.

Based on by 'django-admin startproject' using Django 2.1.2.

For more information on this file, see
https://docs.djangoproject.com/en/2.1/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/2.1/ref/settings/
"""

import os
from pathlib import Path
import dj_database_url

from django.urls.conf import path

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = Path(__file__).resolve().parent.parent

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/2.1/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'c8de103b-1098-4694-b0be-f02d54d2ede9'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['evapp1.azurewebsites.net', 'localhost', '13.51.86.242']

# Application references
# https://docs.djangoproject.com/en/2.1/ref/settings/#std:setting-INSTALLED_APPS
INSTALLED_APPS = [
    # Add your apps here to enable them
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

      # 3rd party apps
    'taggit',
    'crispy_forms',
    'crispy_bootstrap4',
    # Custom apps
    'photoapp',
    'users',
    'storages',
]

CRISPY_TEMPLATE_PACK = 'bootstrap4'

# Middleware framework
# https://docs.djangoproject.com/en/2.1/topics/http/middleware/
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'config.urls'

# Template configuration
# https://docs.djangoproject.com/en/2.1/topics/templates/
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS':  [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'

# Database
#LOCAL_MODE = os.environ.get('DJANGO_LOCAL', False)

# https://docs.djangoproject.com/en/2.1/ref/settings/#databases
#if LOCAL_MODE:
#    DATABASES = {
#        'default': {
#            'ENGINE': 'django.db.backends.sqlite3',
#            'NAME': BASE_DIR / 'db.sqlite3',
#        }
 #   }
#else:
    # Konfiguracja bazy danych Azure SQL (zmień wartości poniżej na swoje dane)
#DATABASES = {
#        'default': dj_database_url.config(
#        engine='sql_server',
#        host='evserver.database.windows.net',
#        user='nnatalla',
#        password='sysweb1!',
#        database='evdb',
#        port='1433',  # Port domyślny dla SQL Server na Azure
#        options={'driver': 'ODBC Driver 17 for SQL Server'},  # Opcjonalne, zależne od wersji sterownika
#    )
#      }
DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': BASE_DIR / 'db.sqlite3',
        }
}

from .db import *

WHITENOISE_INDEX_FILE = True
WHITENOISE_AUTOREFRESH = True
WHITENOISE_USE_FINDERS = True


# Password validation
# https://docs.djangoproject.com/en/2.1/ref/settings/#auth-password-validators
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
# https://docs.djangoproject.com/en/2.1/topics/i18n/
LANGUAGE_CODE = 'pl-PL'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_L10N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/2.1/howto/static-files/
STATIC_URL = '/static/'
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static')]

STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"
STATIC_ROOT = BASE_DIR / "staticfiles"


TAGGIT_CASE_INSENSITIVE = True

# Django Authentication
LOGIN_URL = 'photo:login'
LOGIN_REDIRECT_URL = 'photo:list'

LOGOUT_REDIRECT_URL = 'photo:list'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

MEDIA_URL = 'media/'
MEDIA_ROOT = BASE_DIR / 'media/'

AWS_ACCESS_KEY_ID = 'AKIA4VIRFZJNPLPUVMGL'
AWS_SECRET_ACCESS_KEY = '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'
AWS_STORAGE_BUCKET_NAME = 'evdb1'
AWS_S3_REGION_NAME = 'eu-north-1'  # np. 'us-west-2'

# Ustawienia dla plików statycznych
AWS_S3_CUSTOM_DOMAIN = f'evdb1.s3.amazonaws.com'
STATIC_URL = f'https://{AWS_S3_CUSTOM_DOMAIN}/static/'
STATICFILES_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'

# Ustawienia dla mediów
MEDIA_URL = f'https://{AWS_S3_CUSTOM_DOMAIN}/media/'
DEFAULT_FILE_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'
