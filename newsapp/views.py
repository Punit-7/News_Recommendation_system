from django.shortcuts import render,get_object_or_404
from .models import Data

# Create your views here.

def home(request):

    obj = Data.objects.all()

    return render(request,'home.html',{'obj': obj})


def content(request,id):

    obj = get_object_or_404(Data,pk=id)

    return render(request,'content.html',{'obj': obj})