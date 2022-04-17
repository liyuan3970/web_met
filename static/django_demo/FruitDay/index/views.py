from django.http import HttpResponse
from django.shortcuts import render, redirect
from .models import *
from .forms import *

# Create your views here.
def index_views(request):
    return render(request,'index.html')

def login_views(request):
    if request.method == 'GET':
        #获取请求源地址
        url = request.META.get('HTTP_REFERER','/')

        #判断session中是否有登录信息(uphone)
        if 'uphone' in request.session:
            # 有登录信息,直接重定向到url位置处
            return redirect(url)
        else:
            # 继续向下判断cookie
            if 'uphone' in request.COOKIES and 'upwd' in request.COOKIES:
                #cookie中有登录信息的,继续向下判断

                #从cookie中获取相应的数据
                uphone = request.COOKIES['uphone']
                upwd = request.COOKIES['upwd']
                #去数据库中判断uphone和upwd是否正确
                users = Users.objects.filter(uphone=uphone,upwd=upwd)
                if users:
                    #cookies中的数据是正确的
                    #将uphone的值保存进session,重定向回url
                    request.session['uphone'] = uphone
                    return redirect(url)
                else:
                    #cookies中的数据是错误的,删除登录信息,重新回到登录页面
                    form = LoginForm()
                    resp = render(request,'login.html',locals())
                    resp.delete_cookie('uphone')
                    resp.delete_cookie('upwd')
                    resp.set_cookie('url',url)
                    return resp
            else:
                #cookie中也没有登录信息,则去往登录页面
                #将请求源地址url封装到cookie中
                form = LoginForm()
                resp = render(request, 'login.html', locals())
                resp.set_cookie('url',url)
                return resp



    else:
        #接收uphone和upwd
        uphone = request.POST['uphone']
        upwd = request.POST['upwd']
        #验证登录操作
        users=Users.objects.filter(uphone=uphone,upwd=upwd)
        if users:
            #登录成功的业务处理
            #将登录uphone值保存进session
            request.session['uphone'] = uphone
            #判断是否要记住密码
            #创建响应对象
            url = request.COOKIES.get('url','/')
            resp = redirect(url) #从哪来回哪去
            if 'isSave' in request.POST:
                max_age = 60 * 60 * 24 * 365
                resp.set_cookie('uphone',uphone,max_age)
                resp.set_cookie('upwd',upwd,max_age)
            return resp
        else:
            form = LoginForm()
            return render(request,'login.html',locals())

def register_views(request):
    if request.method == 'GET':
        return render(request, 'register.html')
    else:
        uphone = request.POST.get('uphone')
        #验证uphone在数据库中是否存在
        users=Users.objects.filter(uphone=uphone)
        if users:
            return render(request,'register.html',{'errMsg':'电话号码已经存在'})
        #继续接收其他数据
        upwd = request.POST.get('upwd')
        uname = request.POST.get('uname')
        uemail = request.POST.get('uemail')
        Users.objects.create(
            uphone=uphone,
            upwd = upwd,
            uname = uname,
            uemail = uemail
        )
        return redirect('/')



def register01_views(request):
    Users.objects.create(
        uphone = '13912345678',
        upwd = '123456',
        uname = '隔壁老王',
        uemail = 'wangwc@163.com'
    )
    return HttpResponse("注册成功")

