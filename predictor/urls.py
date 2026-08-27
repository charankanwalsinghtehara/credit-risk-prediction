from django.urls import path
from . import views


urlpatterns = [

    path(
        "",
        views.home,
        name="home"
    ),

    path(
        "predict/",
        views.predict_risk,
        name="predict_risk"
    ),

    path(
        "history/",
        views.history,
        name="history"
    ),

    path(
        "clear-history/",
        views.clear_history,
        name="clear_history"
    ),
]