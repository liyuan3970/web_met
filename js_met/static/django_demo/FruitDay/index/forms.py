from django import forms
from .models import *

class LoginForm(forms.ModelForm):
  class Meta:
    model = Users
    fields = ["uphone","upwd"]
    labels = {
      "uphone": '手机号',
      "upwd": '密码'
    }
    #指定小部件
    widgets = {
      "uphone": forms.TextInput(
        attrs={
          'class':'uinput'
        }
      ),
      "upwd": forms.PasswordInput(
        attrs = {
          'class': 'uinput',
          'placeholder':'请输入6-20位字符'
        }
      )
    }










