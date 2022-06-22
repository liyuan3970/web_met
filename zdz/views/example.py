from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

from ..forms import TestForm


# 只允许post请求
@require_http_methods(["GET", "POST"])
def test(request):
    # POST接收表单数据，GET接收url参数
    valid_data = TestForm(request.POST)
    if valid_data.is_valid():
        clean_date = valid_data.cleaned_data
        print(clean_date)
    return JsonResponse({"msg": valid_data.errors}, json_dumps_params={"ensure_ascii": False})
