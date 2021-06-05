from django.shortcuts import render,get_object_or_404,redirect
from .models import Data,rate
import random

# Create your views here.

def home(request):
    count = Data.objects.all().count()
    slice = random.random() * (count - 10)
    obj = Data.objects.all()[slice: slice + 13]

    Context = {'object2': obj[0], 'object3': obj[1], 'object4': obj[3], 'object5': obj[4], 'object6': obj[5],
               'object7': obj[6], 'object8': obj[7], 'object9': obj[8], 'object10': obj[9], 'object11': obj[10],
               'object12': obj[11], 'object13': obj[12]}

    return render(request,'home.html',Context)


def content(request,id):

    obj = get_object_or_404(Data,pk=id)

    return render(request,'content.html',{'obj': obj})
def getRating(request):
    if request.method=='POST':
        rating=request.POST['rating']
        articleId = request.POST['articleId']
        userId = request.POST['userId']
    r=rate(rating=rating,articleId=articleId,userId=userId)
    r.save()
    return redirect('/')
