from django.db import models

# Create your models here.
class Data(models.Model):
    Title = models.CharField(max_length=1000)
    Author = models.CharField(max_length=1000)
    Content = models.TextField()
    URL = models.URLField(max_length=2000)

class rate(models.Model):
    rating=models.IntegerField()
    articleId=models.IntegerField()
    userId=models.IntegerField()

class comments(models.Model):
    comment=models.TextField()
    articleId=models.IntegerField()
    userName=models.CharField(default="", max_length=1000)
