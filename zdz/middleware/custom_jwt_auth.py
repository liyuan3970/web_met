from django.contrib.auth.backends import ModelBackend

from zdz.common.constant.enum import ResponseCodeEnum
from zdz.common.exceptions import BusinessException
from ..models import User


class CustomJWTAuth(ModelBackend):

    def authenticate(self, request, username=None, password=None, **kwargs):
        try:
            try:
                user = User.objects.get(username=username)
            except Exception:
                raise BusinessException(ResponseCodeEnum.ACCOUNT_NOT_EXISTED)

            if user.check_password(password):
                return user
            else:
                raise BusinessException(ResponseCodeEnum.PASSWORD_ERROR)

        except Exception as e:
            raise e
