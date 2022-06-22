import logging

from django.middleware.common import MiddlewareMixin
from django.shortcuts import render

logger = logging.getLogger('django')


class ExceptionMiddleware(MiddlewareMixin):

    def process_exception(self, request, exception):
        if isinstance(exception, Exception):
            logger.error(exception)
            return render(request, 'error.html')
