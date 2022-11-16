from django.http import HttpResponse
from rest_framework.decorators import api_view
from weasyprint import HTML


@api_view(["post"])
def pdf(request):
    # 接收前台数据
    document = request.data["document"]

    pdf_file = HTML(string=document)

    pdf_file = pdf_file.write_pdf()
    return HttpResponse(pdf_file, content_type="application/pdf")
