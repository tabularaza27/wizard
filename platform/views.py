from django.shortcuts import render, redirect

def index(request):
    return render(request, 'home.html')

def start(request):
    return render(request, 'start.html')
