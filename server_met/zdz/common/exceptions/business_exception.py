from rest_framework.exceptions import APIException

from zdz.common.constant.enum import ResponseCodeEnum


class BusinessException(APIException):

    def __init__(self, enum: ResponseCodeEnum):
        self.detail = {
            "code": enum.code,
            "err_msg": enum.desc
        }
