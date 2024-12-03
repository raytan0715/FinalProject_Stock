from django.urls import path
from . import views

urlpatterns = [
    path('', views.fetch_stock_data, name='fetch_stock_data'),
]
