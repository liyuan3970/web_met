from django.http import HttpResponse
from rest_framework import permissions
from rest_framework.decorators import api_view, permission_classes
from weasyprint import HTML

from ..models.doucument_model import *


@api_view(["post"])
@permission_classes([permissions.IsAuthenticated])
def pdf(request):
    # 接收前台数据
    document = request.data["document"]

    pdf_file = HTML(string=document)

    pdf_file = pdf_file.write_pdf(presentational_hints=True)
    return HttpResponse(pdf_file, content_type="application/pdf")


@api_view(["post"])
@permission_classes([permissions.IsAuthenticated])
def preview_save(request):
    # 接收前台数据
    document = request.data["document"]
    item = request.data["item"]
    items = request.data["items"]
    types = request.data["types"]
    name = request.data["name"]
    year = request.data["year"]
    data = request.data["data"]
    unity = request.data["unity"]
    context = {
        "id": item,
        "data": data
    }
    # 数据存储
    obj = SelfDefine.objects.create(
        types=types,
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
@permission_classes([permissions.IsAuthenticated])
def preview_all(request):
    # 接收前台数据
    document = request.data["document"]
    items = request.data["items"]
    types = request.data["types"]
    # name = request.POST["name"]
    name = request.POST.getlist("name")
    #name = request.query_params.getlist("name")
    year = request.data["year"]
    data = request.POST.getlist('data',[])
    unity = request.data["unity"]

    print(type(name),name[0].split(),len(data),type(data),len(name))
    for i in range(len(data)):
        context = {
            "id": i,
            "data": data[i]
        }
        obj = SelfDefine.objects.create(
            types=types,
            name=name[i],
            item=items,
            year=year,
            unity=unity,
            data=context,
            create_user=0,
            update_user=0
        )
    print("所有数据保存成功")
    # 数据存储
    # obj = SelfDefine.objects.create(
    #     types=types,
    #     name=name,
    #     item=items,
    #     year=year,
    #     unity=unity,
    #     data=context,
    #     create_user=0,
    #     update_user=0
    # )
    # print("数据保存成功")
    pdf_file = HTML(string=document)
    pdf_file = pdf_file.write_pdf(presentational_hints=True)
    return HttpResponse(pdf_file, content_type="application/pdf")


@api_view(["post"])
@permission_classes([permissions.IsAuthenticated])
def history_save(request):
    # 接收前台数据
    document = request.data["document"]
    # 查询对应数据库字段的数据再返回

    pdf_file = HTML(string=document)

    pdf_file = pdf_file.write_pdf(presentational_hints=True)
    return HttpResponse(pdf_file, content_type="application/pdf")
