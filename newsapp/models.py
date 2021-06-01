from django.db import models

# Create your models here.
class Data(models.Model):
    Title = models.CharField(max_length=1000)
    Author = models.CharField(max_length=1000)
    Content = models.TextField()
    URL = models.URLField(max_length=2000)

