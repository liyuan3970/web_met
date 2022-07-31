import json
import logging

from rest_framework import status
from rest_framework.views import Response
from rest_framework.views import exception_handler

logger = logging.getLogger("django")


def custom_exception_handler(exc, context):
    response = exception_handler(exc, context)

    if response is None:
        logger.error(exc)
        return Response({"code": status.HTTP_500_INTERNAL_SERVER_ERROR, "err_msg": "服务器内部错误"}, exception=True)
    else:
        if isinstance(response.data, dict):
            code = response.data.get("code", None)
            if code and isinstance(code, int):
                return Response({"code": response.data.get("code"), "err_msg": response.data.get("err_msg")},
                                exception=True)
            else:
                return Response({"code": response.status_code, "err_msg": json.dumps(response.data)}, exception=True)
        else:
            return Response({"code": response.status_code, "err_msg": str(response.data)}, exception=True)
