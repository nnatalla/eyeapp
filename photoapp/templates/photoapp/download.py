# views.py

from django.http import HttpResponse
from django.core.files.storage import default_storage
import mimetypes
import os

def download_file(request, file_path):
    # Pobierz pełną ścieżkę do pliku na serwerze
    file_path = default_storage.path(file_path)

    # Sprawdź, czy plik istnieje
    if not default_storage.exists(file_path):
        return HttpResponse("Plik nie istnieje", status=404)

    # Ustal MIME type pliku
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        mime_type = 'application/octet-stream'

    # Otwórz plik binarny
    with open(file_path, 'rb') as file:
        response = HttpResponse(file.read(), content_type=mime_type)
        # Ustaw nagłówki, aby przeglądarka zrozumiała, że to plik do pobrania
        response['Content-Disposition'] = f'attachment; filename={os.path.basename(file_path)}'
        return response
