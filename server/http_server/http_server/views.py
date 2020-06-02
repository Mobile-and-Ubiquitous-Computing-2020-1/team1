import os

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def push(request):
    dirname = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '../../intermediate-features')
    filename = 'intermediates'
    fullname = os.path.join(dirname, filename)

    os.makedirs(dirname, exist_ok=True)

    with open(fullname, 'wb') as output:
        output.write(request.read())
    return JsonResponse({'success': True})
