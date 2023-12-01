'''Photoapp URL patterns'''

from django.urls import path

from .views import (
    PhotoListView,
    PhotoTagListView,
    PhotoDetailView,
    PhotoCreateView,
    PhotoUpdateView,
    PhotoDeleteView,
    PhotoProcessView,
    PhotoProcessedView,
    PhotoBaseView,
    PhotoBaseDatailView,
    PhotoNewView,
    PhotoNewDetailView,
    PhotoAllView,
    AddNewsView,
    PhotoMineView
    

)

app_name = 'photo'

urlpatterns = [
    path('', PhotoListView.as_view(), name='list'),

    path('tag/<slug:tag>/', PhotoTagListView.as_view(), name='tag'),

    path('photo/<int:pk>/', PhotoDetailView.as_view(), name='detail'),

    path('photo/create/', PhotoCreateView.as_view(), name='create'),

    path('photo/<int:pk>/update/', PhotoUpdateView.as_view(), name='update'),

    path('photo/<int:pk>/delete/', PhotoDeleteView.as_view(), name='delete'),

    path('photo/<int:pk>/process/', PhotoProcessView.as_view(), name='process'),

    path('photo/<int:pk>/processed/', PhotoProcessedView.as_view(), name='processed'),

    path('photo/basephotos/', PhotoBaseView.as_view(), name='basephotos'),

    path('photo/mine/', PhotoMineView.as_view(), name='mine'),

    path('photo/<int:pk>/basedetail/', PhotoBaseDatailView.as_view(), name='basedetail'),

    path('photo/newphotos/', PhotoNewView.as_view(), name='newphotos'),

    path('photo/<int:pk>/newdetail/', PhotoNewDetailView.as_view(), name='newdetail'),

    path('photo/add_news/', AddNewsView.as_view(), name='addnews'),

    path('photo/all/', PhotoAllView.as_view(), name='all'),

]
