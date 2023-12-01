'''Photo app Forms'''


from django import forms
from .models import News, Photo

class NewsForm(forms.ModelForm):
    class Meta:
        model = News
        fields = ['title', 'content']
        widgets = {
            'title': forms.TextInput(attrs={'style': 'color: lightgreen; font-weight: bold;'}),
            'content': forms.Textarea(attrs={'style': 'color: lightgreen; font-weight: bold;'}),
        }
        labels = {
            'title': 'Tytuł',
            'content': 'Treść',
        }

class PhotoForm(forms.ModelForm):
    class Meta:
        model = Photo
        fields = ['title', 'description', 'image', 'tags', 'processed_image']
        labels = {
            'title': 'Tytuł',
            'description': 'Opis',
            'image': 'Zdjęcie',
            'tags': 'Tagi',
            'processed_image': 'Przetworzone zdjęcie',  
        }
        help_texts = {
            'tags': 'Podaj tagi, oddzielając je przecinkami.',  
            'processed_image': 'Opcjonalnie możesz dodać już przetworzony obraz.', 
        }


class UpdateForm(forms.ModelForm):
    class Meta:
        model = Photo
        fields = ['title', 'description', 'tags', 'processed_image']
        labels = {
            'title': 'Tytuł',
            'description': 'Opis',
            'tags': 'Tagi',
            'processed_image': 'Przetworzone zdjęcie',  
        }
        help_texts = {
            'tags': 'Podaj tagi, oddzielając je przecinkami.',  
            'processed_image': 'Opcjonalnie możesz dodać już przetworzony obraz.', 
        }


