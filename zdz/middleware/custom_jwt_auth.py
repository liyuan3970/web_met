from django.contrib.auth.backends import ModelBackend
from django.db.models import Q
from rest_framework import serializers

from ..models import *


class CustomJWTAuth(ModelBackend):

    def authenticate(self, request, username=None, password=None, **kwargs):
        # print(request.data) 参考请求的其他数据
        # print(request.data['demo']) 比如说key是demo的数据用来做你要的数据校验
        try:
            # 小编这里添加了一个手机验证，如果需要其他验证再加就ok了
            try:
                user = Users.objects.get(Q(username=username) | Q(mobile=username))
            except Exception:
                raise serializers.ValidationError({'': '账号没有注册'})

            if user.check_password(password):
                return user
            else:
                # 如果不想密码登录也可以验证码在这里写
                # 这里做验证码的操作
                raise serializers.ValidationError({'': '密码错误'})

        except Exception as e:
            raise e
