import os
import json
from moviepy.editor import *
import moviepy.editor as mp
from os.path import dirname, join
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.http.response import StreamingHttpResponse
from streamapp.camera import VideoCamera
from django.template.loader import render_to_string
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from django.template import loader, Context
from json import dumps 
from django.template.defaulttags import register
from threading import Thread
import requests
import speech_recognition as sr
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
import base64
import gtts
from deep_translator import GoogleTranslator
from moviepy.editor import *
from moviepy.editor import *
import pdfplumber 

from static.cannyeval_v3 import *
evaluatorA = CannyEval() 
#path='/staticfiles'
cloud_config= {
        'secure_connect_bundle': join(dirname(__file__), "secure-connect-database.zip")
}
#auth_provider = PlainTextAuthProvider('Datauser', 'database@1')
#cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
#session = cluster.connect()
d= {'hi': 'hindi','en': 'english','ta': 'tamil','kn': 'kannada', 'te': 'telugu','ml' : 'malayalam','mr': 'marathi','gu': 'gujarati','pa': 'punjabi', 'cs': 'czech', 'hu': 'hungarian', 'sq': 'albanian', 'mg': 'malagasy', 'yo': 'yoruba', 'gd': 'scots gaelic', 'mt': 'maltese', 'bn': 'bengali', 'st': 'sesotho', 'az': 'azerbaijani', 'hy': 'armenian', 'hr': 'croatian', 'am': 'amharic', 'pt': 'portuguese', 'lt': 'lithuanian', 'tl': 'fil ipino', 'mn': 'mongolian', 'ceb': 'cebuano', 'lv': 'latvian', 'fil': 'Filipino', 'my': 'myanmar (burmese)', 'hmn': 'hmong', 'ar': 'arabic', 's u': 'sundanese', 'cy': 'welsh', 'no': 'norwegian', 'ja': 'japanese', 'uk': 'ukrainian', 'el': 'greek', 'sw': 'swahili', 'fr': ' french', 'ky': 'kyrgyz', 'kk': 'kazakh',  'km': 'khmer', 'sv': 'swedish', 'id': 'indonesian', 'fa': 'persian', 'ku': 'kurdish (k urmanji)', 'zh-tw': 'chinese (traditional)', 'be': 'belarusian', 'sr': 'serbian', 'ht': 'haitian creole',  'de': 'german',  'ps': 'pashto', 'ha': 'hausa', 'ru': 'russian',  'gl': 'galician', 'co': 'corsican', 'so': 'somali', 'th': 'tha i', 'uz': 'uzbek', 'ms': 'malay', 'da': 'danish', 'ny': 'chichewa', 'sn': 'shona', 'nl': 'dutch', 'lo': 'lao', 'fy': 'frisian ', 'sl': 'slovenian', 'fi': 'finnish', 'ig': 'igbo', 'et': 'estonian', 'tr': 'turkish', 'bg': 'bulgarian', 'eu': 'basque', 'haw': 'hawaiian', 'sk': 'slovak', 'si': 'sinhala', 'ne': 'nepali', 'ro': 'romanian', 'ca': 'catalan', 'xh': 'xhosa', 'ga': 'irish', 'lb': 'luxembourgish', 'jw': 'javanese', 'iw': 'hebrew', 'zh-cn': 'chinese (simplified)', 'tg': 'tajik', 'ko': 'korean', 'ur': 'urdu', 'it': 'italian', 'zu': 'zulu',  'mk': 'macedonian', 'sm': 'samoan', 'es': 'spanish', 'sd': 'sindhi',  'pl': 'polish', 'vi': 'vietnamese',  'ka': 'georgian', 'yi': 'yiddish', 'la': 'latin', 'eo': 'esperanto', 'af': 'afrikaans', 'is': 'icelandic', 'bs': 'bosnian', 'mi': ' maori', 'zh': 'chinese', 'he': 'Hebrew'}

#row = session.execute("select release_version from system.local").one()
#if row:
 #   print(row[0])
#else:
 #   print("An error occurred.")
#session.set_keyspace('data')

############

ids=[]


def index(request):
    return render(request, 'streamapp/input.html')
def quiz(request):
    return render(request, 'streamapp/newquiz.html')
    #return render(request, 'streamapp/copyquiz.html',{'a':3,'q1':{"QUESTION 1 Who developed the Python language?":['.py','.p','.c','.java'],"Which one of the following is the correct extension of the Python file?":['Zim Den','Guido van Rossum','Niene Stom','Wick van Rossum','.py','.python','.p','.java'],"QUESTION 1 Who developed the Python language?":['.py','.p','.c','.java'],"Which one of the following is the correct extension of the Python file?":['Zim Den','Guido van Rossum','Niene Stom','Wick van Rossum','.py','.python','.p','.java'],"QUESTION 1 Who developed the Python language?":['.py','.p','.c','.java'],"Which one of the following is the correct extension of the Python file?":['Zim Den','Guido van Rossum','Niene Stom','Wick van Rossum','.py','.python','.p','.java']}})

def take(request):
 
    global ids
    global d
 
      #  futures = []
     #   query = "SELECT id FROM teacher"
       # futures.append(session.execute_async(query))
       # l = []
       # for future in futures:
       #     rows = future.result()
       #     l.append(str(rows[0].id))
      #  if user in l:
      #      c = "Sorry" + " " + "Please Login Account Already Exists"
       #     return render(request, 'streamapp/input.html', {'resultt': c,'d':d})
    #    else:
    ##        insert_statement = session.prepare("INSERT INTO teacher (id,psw) VALUES (?,?)")
      #      session.execute(insert_statement, [user, b])
       #     c =user.capitalize() 
 
      #  futures.append(session.execute_async(query))
       # l = []
        #for future in futures:
         #   rows = future.result()
          #  l.append(str(rows[0].id))
        #if user in l:
         #   c = "Sorry" + " " + "Please Login Account Already Exists"
          #  return render(request, 'streamapp/input.html', {'resultt': c,'d':d})
        #else:
         #   insert_statement = session.prepare("INSERT INTO userdata (id,pass) VALUES (?,?)")
          #  session.execute(insert_statement, [user, b])
           # c =user.capitalize() 
    return render(request, 'streamapp/landt.html', {'user': "Raj",'d':'d'})

def add(request):
    global dpass
    global ids
    global user
    global d
    dpass=""
  #  st="request.POST.get('st')"
    #user=request.POST.get('user')
    #passl=request.POST.get('pass')
    return render(request, 'streamapp/dashboard.html')
        

Test=False
frame1=""
def gen(camera):
    global frame1
    global Test
    global user
    while True:
        if Test==True:
            print(frame1)
          #  session.set_keyspace('data')
          #  insert_statement = session.prepare("INSERT INTO userdata (id,marks) VALUES (?,?)")
         #   session.execute(insert_statement, [user,str(frame1)])
            print("finaly break")
            break
        frame,frame1 = camera.get_frame()
        
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_fee(request):
        global Test
        print("I AM THERE")
        Test=True
        c=request.POST.get('num1')

        return render(request, 'streamapp/marks.html', {'c': a[:-1],'c1':['Number of Times Mouth Opened','Number of Times Head Up','Number of Times Head Down','Number of Times Head Left','Number of Times Head Right','Number of Times Left the Test'],'s':a1})
def correction(request):
        global Test
       
        Test=True
        
        c=(request.POST.get('testName'))
        print(c)    
        num_q=1
        teach_ans=["All the animals in the world will go under hibernation in winter season"]
        student_ans=[c]
        a=(frame1).split(" ")
        for i in range(len(a)):
            a[i]=int(a[i])
        print("trust",a,student_ans)
        report=calc(num_q,teach_ans,student_ans)
        op={"name": "Raj","total": 48,"wrong":2,"correct": 8,"trust":str(a[:-1])}
        #report=5
       
        return render(request, 'streamapp/result.html',{"trust":a[:-1],"tust_score":a[-1],"mark":report[0],"student_ans":student_ans[0],"op":op})
def calc(num_q,teach_ans,student_ans):
    dj=dict()
    #{"1":["Sunrise and Sunset are good to see but we never can't expect them to see at the same time", [["Both Sunrise and Sunset are good to see"]]],"2":["all the animals in the world will go under hibernation in winter season", [["Animals will undergo hibernation in summer season"]]]}
    for i in range(num_q):
        dj[str(i+1)]=[str(teach_ans[i]),[[str(student_ans[i])]]]
    
    #dj = json.load(open('./minimal-data-4.json'))
    report = evaluatorA.report_card(data_json=dj, max_marks=10, relative_marking=False, integer_marking=False, json_load_version="v2")
    out = eval(report.to_json()) 
    print("Report of the evaluations")
    print(out)
    anss=[]
    for i in out:
        #print(out[i]['0'])
        anss.append(out[i]['0'])
    
    return anss

def video_feed(request):
    print(Test)
    if not(Test):
        return StreamingHttpResponse(gen(VideoCamera()),
                        content_type='multipart/x-mixed-replace; boundary=frame')
    print("video freed")
    return render(request, 'streamapp/wrong.html', {'res': "bryeu", 'data': False})
global l
l=[]
    
