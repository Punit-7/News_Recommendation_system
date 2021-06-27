from django.shortcuts import render
from .models import Data

# Create your views here.

def home(request):
    obj2=Data.objects.get(id=2)
    obj3= Data.objects.get(id=3)
    obj4 = Data.objects.get(id=4)
    obj5= Data.objects.get(id=5)
    obj6 = Data.objects.get(id=6)
    obj7 = Data.objects.get(id=7)
    obj8 = Data.objects.get(id=8)
    obj9 = Data.objects.get(id=9)
    obj10 = Data.objects.get(id=10)
    obj11 = Data.objects.get(id=11)
    obj12 = Data.objects.get(id=12)
    obj13 = Data.objects.get(id=13)

    context={'object2':obj2,'object3':obj3,'object4':obj4,'object5':obj5,'object6':obj6,'object7':obj7,'object8':obj8,'object9':obj9,'object10':obj10,'object11':obj11,'object12':obj12,'object13':obj13}

    return render(request,'home.html',context)