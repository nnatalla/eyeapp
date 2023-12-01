from django.urls import path

from django.contrib.auth.views import LogoutView
from django.views.generic.base import TemplateView

from .views import NewsListView, SignUpView, CustomLoginView

app_name = 'user'

urlpatterns = [
    path('signup/', SignUpView.as_view(), name='signup'),
    path('login/', CustomLoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('base/',TemplateView.as_view(template_name='base.html'), name='base'),
    path('actual/',NewsListView.as_view(template_name='actual.html'), name='actual'),
    path('about/',TemplateView.as_view(template_name='about.html'), name='about'),
    path('eye/',TemplateView.as_view(template_name='eye.html'), name='eye'),
    path('algorithm/',TemplateView.as_view(template_name='algorithm.html'), name='algorithm'),
    path('contact/',TemplateView.as_view(template_name='contact.html'), name='contact'),

]