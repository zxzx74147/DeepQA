from django.http import HttpResponse
from django.shortcuts import render_to_response

from .chatbotmanager import ChatbotManager
import logging
logger = logging.getLogger(__name__)

def chat(request):
    if 'question' in request.GET:
        question=request.GET['question']
        answer = ChatbotManager.callBot(question)
        if not answer:
            answer = 'Error: Try a shorter sentence'
        logger.info(' {} -> {}'.format( question, answer))
        return HttpResponse(answer)
    return HttpResponse('no question')