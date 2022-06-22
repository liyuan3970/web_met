import logging

from django.middleware.common import MiddlewareMixin
from django.shortcuts import render


class ExceptionMiddleware(MiddlewareMixin):

    def process_exception(self, request, exception):
        if isinstance(exception, Exception):
            logging.error(exception)
            return render(request, 'error.html')
