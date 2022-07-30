from django.contrib.auth.backends import ModelBackend
from rest_framework import serializers

from ..models import User


class CustomJWTAuth(ModelBackend):

    def authenticate(self, request, username=None, password=None, **kwargs):
        try:
            try:
                user = User.objects.get(username=username)
            except Exception as e:
                raise serializers.ValidationError({'': '账号没有注册'})

            return user
            # if user.check_password(password):
            #     return user
            # else:
            #     raise serializers.ValidationError({'': '密码错误'})

        except Exception as e:
            raise e
