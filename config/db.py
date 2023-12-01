# To use Neon with Django, you have to create a Project on Neon and specify the project connection settings in your settings.py in the same way as for standalone Postgres.

DATABASES = {
  'default': {
    'ENGINE': 'django.db.backends.postgresql',
    'NAME': 'evdb',
    'USER': 'nnatalla',
    'PASSWORD': 'CNmqID6MrPg3',
    'HOST': 'ep-round-hat-67932505.eu-central-1.aws.neon.tech',
    'PORT': '5432',
    'OPTIONS': {'sslmode': 'require'},
  }
}