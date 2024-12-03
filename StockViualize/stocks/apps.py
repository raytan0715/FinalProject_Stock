from django.apps import AppConfig

class StocksConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'  # Django 默認的主鍵類型
    name = 'stocks'  # 應用名稱，對應目錄名稱
