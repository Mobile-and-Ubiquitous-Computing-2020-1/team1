import os

from django.http import HttpResponse, JsonResponse, HttpResponseNotAllowed
from django.views.decorators.csrf import csrf_exempt
from django.utils.encoding import smart_str

@csrf_exempt
def push(request):
    if request.method != 'POST':
        return HttpResponseNotAllowed('POST')

    dirname = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '../../intermediate-features')
    filename = request.headers.get("Content-Disposition").split("filename=")[1].replace('"', '')
    fullname = os.path.join(dirname, filename)

    os.makedirs(dirname, exist_ok=True)

    with open(fullname, 'wb') as output:
        output.write(request.read())
    return JsonResponse({'success': True})

def pull(request):
    if request.method != 'GET':
        return HttpResponseNotAllowed('GET')

    TFLITE_MODEL_DIR = '../../client/app/src/main/assets'
    filename = 'mobilenet_v1.tflite'
    fullname = os.path.join(TFLITE_MODEL_DIR, filename)
    with open(os.path.join(TFLITE_MODEL_DIR, filename), 'rb') as f:
        response = HttpResponse(f, content_type='application/octet-stream')
        return response
