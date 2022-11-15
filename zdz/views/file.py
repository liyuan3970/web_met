from django.http import HttpResponse
from rest_framework.decorators import api_view
from weasyprint import HTML


@api_view(["post"])
def pdf(request):
    # 接收前台数据
    pdf_blob = request.data["pdf_blob"]

    pdf = HTML(string='''
    <h1>2323143</h1>
    <p>12312321<p>
''')
    pdf_file = pdf.write_pdf()
    return HttpResponse(pdf_file, content_type="application/pdf")
