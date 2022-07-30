from rest_framework import status
from rest_framework.views import Response
from rest_framework.views import exception_handler


def custom_exception_handler(exc, context):
    message = ""
    response = exception_handler(exc, context)

    for index, value in enumerate(response.data):
        if index == 0:
            key = value
            value = response.data[key]

            if isinstance(value, str):
                message = value
            else:
                message = key + value[0]

    if response is None:
        return Response({"err_msg": "服务器错误"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR, exception=True)
    else:
        return Response({"err_msg": message}, status=response.status_code, exception=True)
