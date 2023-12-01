from django.views.generic import CreateView

from django.contrib.auth import authenticate, login

from django.contrib.auth.views import LoginView

from django.views.generic import ListView, View

from django.contrib.auth.forms import UserCreationForm

from django.urls import reverse_lazy

from photoapp.models import News

from django.shortcuts import render

from datetime import timedelta

from django.utils import timezone

from django.http import JsonResponse




class SignUpView(CreateView):

    template_name = 'users/signup.html'
    
    form_class = UserCreationForm

    success_url = reverse_lazy('photo:list')

    def form_valid(self, form):
        to_return = super().form_valid(form)

        user = authenticate(
            username=form.cleaned_data["username"],
            password=form.cleaned_data["password1"],
        )

        login(self.request, user)

        return to_return
    
class CustomLoginView(LoginView):
    
    template_name = 'users/login.html'

class NewsListView(View):
    template_name = 'users/actual.html'

    def get(self, request):
        num_to_display = int(request.GET.get('num_to_display', 3))
        date_filter = int(request.GET.get('date_filter', 3))

        start_date = timezone.now() - timedelta(days=date_filter)

        latest_news = News.objects.filter(date_published__gte=start_date).order_by('-date_published')[:num_to_display]

        request.session['num_to_display'] = num_to_display
        request.session['date_filter'] = date_filter

        context = {
            'latest_news': latest_news,
            'selected_num': num_to_display,
            'selected_date_filter': date_filter,
            'display_options': [3, 5, 10],
            'date_filter_options': [1, 3, 7, 30],
        }

        return render(request, self.template_name, context)

    def ajax_get(self, request):
        num_to_display = request.GET.get('num_to_display', request.session.get('num_to_display', 3))
        date_filter = request.GET.get('date_filter', request.session.get('date_filter', 3))

        start_date = timezone.now() - timedelta(days=int(date_filter))

        latest_news = News.objects.filter(date_published__gte=start_date)[:int(num_to_display)]

        request.session['num_to_display'] = num_to_display
        request.session['date_filter'] = date_filter

        context = {
            'latest_news': latest_news,
            'selected_num': num_to_display,
            'selected_date_filter': date_filter,
            'display_options': [3, 5, 10],
            'date_filter_options': [1, 3, 7, 30],
        }

        return JsonResponse({'html': render_to_string(self.template_name, context)})
    
