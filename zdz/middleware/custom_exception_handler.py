import json
import logging

from rest_framework import status
from rest_framework.views import Response
from rest_framework.views import exception_handler

logger = logging.getLogger("django")


def custom_exception_handler(exc, context):
    exec_response = exception_handler(exc, context)

    if exec_response is None:
        response = Response({"code": status.HTTP_500_INTERNAL_SERVER_ERROR, "err_msg": "服务器内部错误"},
                            exception=True)
    else:
        if isinstance(exec_response.data, dict):
            code = exec_response.data.get("code", None)
            if code and isinstance(code, int):
                response = Response(
                    {"code": exec_response.data.get("code"), "err_msg": exec_response.data.get("err_msg")},
                    exception=True)
            else:
                response = Response(
                    {"code": exec_response.status_code, "err_msg": json.dumps(exec_response.data, ensure_ascii=False)},
                    exception=True)
        else:
            response = Response({"code": exec_response.status_code, "err_msg": str(exec_response.data)}, exception=True)

    logger.error({
        "code": response.data["code"],
        "err_msg": response.data["err_msg"],
        "exec": exc
    })
    return response
