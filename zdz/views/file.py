from django.db import transaction
from django.http import HttpResponse, JsonResponse
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.parsers import JSONParser
from weasyprint import HTML

from ..models.doucument_model import *
from ..serializers.preview_all_serializer import PreviewAllSerializer


class FileViewSet(viewsets.ViewSet):
    @action(methods=["post"], url_path="pdf", detail=False)
    def pdf(self, request):
        # 接收前台数据
        document = request.data["document"]
        pdf_file = HTML(string=document)
        pdf_file = pdf_file.write_pdf(presentational_hints=True)
        return HttpResponse(pdf_file, content_type="application/pdf")
    @action(methods=["post"], url_path="preview_save", detail=False)
    def preview_save(self, request):
        # 接收前台数据
        document = request.data["document"]
        item = request.data["item"]
        items = request.data["items"]
        document_type = request.data["types"]
        name = request.data["name"]
        year = request.data["year"]
        data = request.data["data"]
        unity = request.data["unity"]
        title = request.data["title"]
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
        # 保存模板
        if title:
            obj = DocumentSelfDefine.objects.create(
                document_type=document_type,
                name=title,
                item=items,
                year=year,
                unity=unity,
                data=context,
                create_user=0,
                update_user=0
            )
        pdf_file = HTML(string=document)
        pdf_file = pdf_file.write_pdf(presentational_hints=True)
        return HttpResponse(pdf_file, content_type="application/pdf")

    # 保存所有文档
    @transaction.atomic()
    @action(methods=["post"], url_path="preview_all", detail=False)
    def preview_all(self, request):
        doc_ser = PreviewAllSerializer(
            data=JSONParser().parse(request),
            context={"user_id": request.user.id}
        )
        doc_ser.is_valid(raise_exception=True)
        doc_ser.save()
        pdf_file = HTML(string=doc_ser.validated_data["document"])
        pdf_file = pdf_file.write_pdf(presentational_hints=True)
        return HttpResponse(pdf_file, content_type="application/pdf")

    @action(methods=["post"], url_path="history_save", detail=False)
    def history_save(self, request):
        # 接收前台数据
        document = request.data["document"]
        # 查询对应数据库字段的数据再返回
        pdf_file = HTML(string=document)
        pdf_file = pdf_file.write_pdf(presentational_hints=True)
        return HttpResponse(pdf_file, content_type="application/pdf")
    @action(methods=["post"], url_path="history_tz", detail=False)
    def history_tz(self, request):
        # 接收前台数据
        info = request.POST.get('info', '')
        if info == 'none':
            unity = "台州市气象局"
            year = 2023
            doc_types = DocumentType.objects.filter(unity=unity).all().values()
            title = list(SelfPlot.objects.values('document_type', 'id'))[-5:]
            type_list = []
            for doc_type in doc_types:
                doc_single = {
                    'name': doc_type['name'],
                    'filelist': []
                }
                doc_all = Document.objects.filter(
                    unity="台州市气象局", year=year, document_type=doc_type['name']).all().values()
                for doc in doc_all:
                    doc_dir = {}
                    doc_dir['type'] = doc['document_type']
                    doc_dir['item'] = doc['item']
                    doc_dir['unity'] = doc['unity']
                    doc_dir['year'] = doc['year']
                    doc_single['filelist'].append(doc_dir)
                type_list.append(doc_single)
            content = {
                'doc_list': type_list,
                'self_title': title
            }
            return JsonResponse(content)
        else:
            fields = info.split("-")
            unity = "台州市气象局"
            item = int(fields[0])
            doc_type = str(fields[1])
            year = int(fields[2])
            hearder = request.POST.get('hearder', '')
            data = Document.objects.filter(
                year=year, unity=unity, document_type=doc_type).all().values()
            version_content = str(data[0]['version_content']).split(",")[0:-1]
            content_list = []
            content_str = hearder + ""
            for i in range(len(version_content)):
                name = version_content[i]
                single_content = ""
                data_self = SelfModule.objects.filter(
                    year=year, name=name, item=item, unity=unity, document_type=doc_type).all().order_by('-create_time').values()
                for j in data_self:
                    if int(j['data']['id']) == i:
                        single_content = j['data']['data']
                        content_list.append(single_content)
                        content_str = content_str + single_content
                        # print("查找数据",j['name'],j['data']['id'],j['types'],j)
                        break
            content = {
                'status': "ok",
                'type_list': version_content,
                'item': item,
                'year': year,
                'type': doc_type,
                'unity': unity,
                'content_list': content_list
            }
            pdf_file = HTML(string=content_str)
            pdf_file = pdf_file.write_pdf(presentational_hints=True)
            return HttpResponse(pdf_file, content_type="application/pdf")

