from django import forms


class TestForm(forms.Form):
    book = forms.CharField(max_length=20)
    number = forms.IntegerField(min_value=0)
    mark = forms.CharField(required=False, max_length=20)
