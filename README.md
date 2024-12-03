.vscode
├── launch.json                  <-Django launch網頁需要
├── seeting.json

FinalProject_stock/
├── StockViualize/                <- Django 主專案
│   ├── StockViualize/            <- 配置模組
│   │   ├── __init__.py
│   │   ├── asgi.py
│   │   ├── settings.py           <- 主配置檔
│   │   ├── urls.py               <- 全局 URL 配置
│   │   └── wsgi.py
│   ├── db.sqlite3                <- SQLite 數據庫文件
│   └── manage.py                 <- Django 命令管理入口
│   ├── stocks/                       <- 應用程式 (app)
│   │   ├── __init__.py               <- 必須存在，告知 Python 這是一個模組
│   │   ├── apps.py                   <- 應用的設定檔
│   │   ├── static/                   <- 靜態文件
│   │   ├── templates/                <- HTML 模板文件
│   │   │   └── stocks/
│   │   │       ├── query.html
│   │   │       └── result.html
│   │   ├── urls.py                   <- 應用的 URL 配置
│   │   └── views.py                  <- 應用邏輯
└── Read.me                       <- 可選的專案說明文件

launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Django",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/StockViualize/manage.py",
            "args": ["runserver"],
            "django": true,
            "console": "integratedTerminal"
        }
    ]
}
# FinalProject_Stock
# FinalProject_Stock
# FinalProject_Stock
