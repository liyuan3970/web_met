from django.db import transaction
from django.http import HttpResponse
from rest_framework.decorators import api_view
from rest_framework.parsers import JSONParser
from weasyprint import HTML

from ..models.doucument_model import *
from ..serializers.preview_all_serializer import PreviewAllSerializer


@api_view(["post"])
def pdf(request):
    # 接收前台数据
    document = request.data["document"]

    pdf_file = HTML(string=document)

    pdf_file = pdf_file.write_pdf(presentational_hints=True)
    return HttpResponse(pdf_file, content_type="application/pdf")


@api_view(["post"])
def preview_save(request):
    # 接收前台数据
    document = request.data["document"]
    item = request.data["item"]
    items = request.data["items"]
    document_type = request.data["types"]
    name = request.data["name"]
    year = request.data["year"]
    data = request.data["data"]
    unity = request.data["unity"]
    context = {
        "id": item,
        "data": data
    }
    # 数据存储
    obj = SelfModule.objects.create(
        document_type=document_type,
        name=name,
        item=items,
        year=year,
        unity=unity,
        data=context,
        create_user=0,
        update_user=0
    )
    print("数据保存成功")

    pdf_file = HTML(string=document)

    pdf_file = pdf_file.write_pdf(presentational_hints=True)
    return HttpResponse(pdf_file, content_type="application/pdf")


# 保存所有文档
@api_view(["post"])
@transaction.atomic()
def preview_all(request):
    doc_ser = PreviewAllSerializer(
        data=JSONParser().parse(request),
        context={"user_id": request.user.id}
    )
    doc_ser.is_valid(raise_exception=True)
    doc_ser.save()
    pdf_file = HTML(string=doc_ser.validated_data["document"])
    pdf_file = pdf_file.write_pdf(presentational_hints=True)
    return HttpResponse(pdf_file, content_type="application/pdf")


@api_view(["post"])
def history_save(request):
    # 接收前台数据
    document = request.data["document"]
    # 查询对应数据库字段的数据再返回
    pdf_file = HTML(string=document)
    pdf_file = pdf_file.write_pdf(presentational_hints=True)
    return HttpResponse(pdf_file, content_type="application/pdf")
