from .base import *

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

# Database
# https://docs.djangoproject.com/en/1.11/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        # 数据库名称
        'NAME': 'tzweb',
        'USER': 'root',
        'PASSWORD': '051219',
        # 'HOST': '192.168.192.5',
        'HOST': '127.0.0.1',
        'PORT': '3306',
    }
}

CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": "redis://192.168.192.5:6379/0",    # /0：代表redis的0号数据库
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
            "PASSWORD": "lq9394"
        }
    }
}


# CACHES = {
#     "default": {
#         "BACKEND": "django_redis.cache.RedisCache",
#         "LOCATION": "redis://192.168.192.5:6379/0",
#         "OPTIONS": {
#             "CLIENT_CLASS": "django_redis.client.DefaultClient",
#             "PASSWORD": ""
#         }
#     }
# }
