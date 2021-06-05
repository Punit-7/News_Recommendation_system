from django.contrib import messages
from django.shortcuts import render, redirect
from django.contrib.auth.models import User,auth
from .models import Profile
# Create your views here.

def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password1']

        user=auth.authenticate(username=username,password=password)
        if user is not None:
            auth.login(request,user)
            return redirect('/')
        else:
            messages.info(request,'Invalid Credentials')
            return redirect('login')

    else:
        return render(request, 'login.html')



def register(request):

    if request.method=='POST':
        first_name=request.POST['first_name']
        last_name = request.POST['last_name']
        email = request.POST['email']
        username = request.POST['username']
        dob= request.POST.get('dob')
        gender=request.POST.get('gender')
        country = request.POST.get('country')
        password1 = request.POST['password1']
        password2 = request.POST['password2']
        if password1 == password2 :
            if User.objects.filter(username=username).exists():
                messages.info(request,'Username Taken')
                return redirect('register')
            elif User.objects.filter(email=email).exists():
                messages.info(request, 'Email Taken')
                return redirect('register')
            else:
                user=User.objects.create_user(first_name=first_name,last_name=last_name,email=email,username=username,password=password1)
                profile = Profile(user=user,dob=dob,gender=gender,country=country)
                profile.save()
        else:
            messages.info(request, 'Password not matching ')
            return redirect('register')

        return redirect('/')
    else:
        return render(request, 'register.html')

def logout(request):
    auth.logout(request)
    return redirect('/')

