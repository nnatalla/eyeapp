'''Photoapp Models'''

from django.db import models

from django.contrib.auth import get_user_model

from taggit.managers import TaggableManager

class Photo(models.Model):
    
    title = models.CharField(max_length=45)
    
    description = models.CharField(max_length=250) 

    created = models.DateTimeField(auto_now_add=True)

    image = models.ImageField(upload_to='media/photos/')

    submitter = models.ForeignKey(get_user_model(), on_delete=models.CASCADE)

    tags = TaggableManager() 

    processed_image = models.ImageField(upload_to='media/processed_images/', blank=True, null=True)
    #processed_image_name = models.CharField(max_length=100, blank=True, null=True)
    def __str__(self):
        return self.title
    

class News(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    date_published = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title


