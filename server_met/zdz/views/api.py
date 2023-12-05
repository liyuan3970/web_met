from django.db import transaction
from django.http import HttpResponse, JsonResponse
from rest_framework import viewsets, permissions
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.parsers import JSONParser
from weasyprint import HTML,Attachment


from ..serializers.zdz_serializer import ZdzSerializer

class ZdzViewSet(viewsets.ViewSet):
    @transaction.atomic()
    @action(methods=["post"], url_path="zdz_demo", detail=False, permission_classes=[permissions.AllowAny])
    def zdz_demo(self, request):
        zdz_ser = ZdzSerializer(
            data=JSONParser().parse(request)
            # context={"user_id": request.user.id}
        )
        zdz_ser.is_valid(raise_exception=True)
        model = zdz_ser.validated_data["model"]
        name = zdz_ser.validated_data["name"]
        print("卧槽")
        context = {
            "model":model,
            "name":name,
            "static":"ok"
        }
        return JsonResponse(context)