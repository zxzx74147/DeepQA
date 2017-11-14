from django.conf.urls import url
from . import views
from . import httpserv

from .chatbotmanager import ChatbotManager

urlpatterns = [
    url(r'^$', views.mainView),
    url(r'^chatapi$', httpserv.chat),
]
