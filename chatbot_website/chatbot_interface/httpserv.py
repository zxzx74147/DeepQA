from django.http import HttpResponse, JsonResponse
from django.shortcuts import render_to_response

from .chatbotmanager import ChatbotManager
import logging
logger = logging.getLogger(__name__)

def chat(request):
    response_data = {}
    if 'question' in request.GET:
        question=request.GET['question']
        answer = ChatbotManager.callBot(question)
        response_data['result'] = '0'
        if not answer:
            answer = 'Error: Try a shorter sentence'
            response_data['result'] = '1'
        response_data['answer'] = answer
        logger.info(' {} -> {}'.format( question, answer))
        return JsonResponse(response_data)
    return JsonResponse(response_data)