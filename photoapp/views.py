'''Photo app generic views'''

from django.shortcuts import get_object_or_404

from django.core.exceptions import PermissionDenied

from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView

from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin

from django.urls import reverse_lazy

from .models import Photo, News

from django.views import View

from .processing import process_image

from django.shortcuts import render, redirect

from django.views import View

from django.contrib import messages

from .forms import NewsForm, PhotoForm, UpdateForm

from taggit.models import Tag

from django.db.models import Q

from django.http import JsonResponse

from .models import Photo

from django.views.generic import ListView

from django.db.models import Q

from django.template.loader import render_to_string

from django.http import HttpResponse

from django.shortcuts import get_object_or_404

from taggit.managers import TaggableManager


class PhotoListView(ListView):
    model = Photo
    template_name = 'photoapp/list.html'
    context_object_name = 'photos'
    ordering = ['-created']
    tags = TaggableManager()


    def get_queryset(self):
        query = self.request.GET.get('search', '')
        search_type = self.request.GET.get('searchType', 'title')

        if query:
            if search_type == 'tags':
                return Photo.objects.filter(tags__name__icontains=query)

            elif search_type == 'submitter':

                return Photo.objects.filter(submitter__username__icontains=query)
            else:
                return Photo.objects.filter(title__icontains=query)
        else:
            return Photo.objects.all()

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['search_query'] = self.request.GET.get('search', '')
        return context

    def render_to_response(self, context, **response_kwargs):
        if self.request.headers.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest':
            return JsonResponse({
                'html': render_to_string(self.template_name, context),
            })
        return super().render_to_response(context, **response_kwargs)
    

class PhotoTagListView(PhotoListView):
    
    template_name = 'photoapp/taglist.html'
    
    def get_tag(self):
        return self.kwargs.get('tag')
    def get_queryset(self):
        return self.model.objects.filter(tags__slug=self.get_tag())
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["tag"] = self.get_tag()
        return context
     

class PhotoDetailView(DetailView):

    model = Photo

    template_name = 'photoapp/detail.html'

    context_object_name = 'photo'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        photo = context['photo']

        processed_image = photo.processed_image
        context['processed_image'] = processed_image

        return context


class PhotoCreateView(LoginRequiredMixin, CreateView):

    model = Photo
    
    form_class = PhotoForm

    template_name = 'photoapp/create.html'
    
    success_url = reverse_lazy('photo:list')

    def form_valid(self, form):

        form.instance.submitter = self.request.user

        title = form.cleaned_data['title']
        if Photo.objects.filter(title=title).exists():
            messages.error(self.request, 'Obraz o tej nazwie już istnieje.')
            return self.form_invalid(form)

        return super().form_valid(form)
    

class UserIsSubmitter(UserPassesTestMixin):

    def get_photo(self):
        return get_object_or_404(Photo, pk=self.kwargs.get('pk'))
    
    def test_func(self):
        
        if self.request.user.is_authenticated:
            return self.request.user == self.get_photo().submitter
        else:
            raise PermissionDenied('Przepraszamy, nie masz dostępu do tej strony.')
        

class PhotoUpdateView(UserIsSubmitter, UpdateView):
    
    template_name = 'photoapp/update.html'

    model = Photo

    form_class = UpdateForm
    
    success_url = reverse_lazy('photo:list')
    def form_valid(self, form):
        
        existing_photo = Photo.objects.exclude(pk=self.object.pk).filter(title=form.cleaned_data['title'])
        if existing_photo.exists():
       
            messages.error(self.request, 'Obraz o tej nazwie już istnieje.')
            return self.form_invalid(form)

        return super().form_valid(form)
    
    
class PhotoDeleteView(UserIsSubmitter, DeleteView):
    
    template_name = 'photoapp/delete.html'

    model = Photo

    success_url = reverse_lazy('photo:list')

    def delete(self, request, *args, **kwargs):

        photo = self.get_object()

        
        photo.image.delete()

        if photo.processed_image:
            photo.processed_image.delete()

        return super().delete(request, *args, **kwargs)



class PhotoProcessView(DeleteView):

    model = Photo
    
    template_name = 'photoapp/process.html'

    

class PhotoProcessedView(DetailView):

    model = Photo

    template_name = 'photoapp/processed.html'

    context_object_name = 'photo'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        photo = context['photo']

        if photo.processed_image:
            photo.processed_image.delete()

        processed_image = process_image(photo.image)

        photo.processed_image = processed_image
        photo.save()

        context['processed_image'] = photo.processed_image
   
        return context


class PhotoBaseView(ListView):
    
    model = Photo     

    template_name = 'photoapp/base.html'

    context_object_name = 'photos'

    ordering = ['-created']


class PhotoBaseDatailView(DetailView):

    model = Photo

    template_name = 'photoapp/base_detail.html'

    context_object_name = 'photo'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        photo = context['photo']

        return context


class PhotoNewView(ListView):
    
    model = Photo     

    template_name = 'photoapp/new.html'

    context_object_name = 'photos'

    ordering = ['-created']


class PhotoNewDetailView(DetailView):

    model = Photo

    template_name = 'photoapp/new_detail.html'

    context_object_name = 'photo'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        photo = context['photo']

        return context

class PhotoAllView(ListView):
    
    model = Photo     

    template_name = 'photoapp/all.html'

    context_object_name = 'photos'

    ordering = ['-created']


class AddNewsView(LoginRequiredMixin, CreateView):
    
    model = News
    
    form_class = NewsForm

    template_name = 'photoapp/add_news.html'

    success_url = reverse_lazy('user:actual')

    def form_valid(self, form):

        form.instance.submitter = self.request.user
        
        return super().form_valid(form)
    
class PhotoMineView(ListView):
    
    model = Photo     

    template_name = 'photoapp/mine.html'

    context_object_name = 'photos'

    ordering = ['created']

