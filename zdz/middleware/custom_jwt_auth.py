from django.contrib.auth.backends import ModelBackend
from rest_framework import serializers

from ..models import UserModel


class CustomJWTAuth(ModelBackend):

    def authenticate(self, request, username=None, password=None, **kwargs):
        print(request.data)
        try:
            try:
                user = UserModel.objects.get(username=username)
            except Exception:
                raise serializers.ValidationError({'': '账号没有注册'})

            return user
            # if user.check_password(password):
            #     return user
            # else:
            #     # 如果不想密码登录也可以验证码在这里写
            #     # 这里做验证码的操作
            #     raise serializers.ValidationError({'': '密码错误'})

        except Exception as e:
            raise e
