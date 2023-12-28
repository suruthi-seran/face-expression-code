from flask import Flask,render_template,Response,request,redirect
from tensorflow.keras.models import model_from_json
import cv2
import numpy as np
import pygame
from time import sleep
import os
from werkzeug.utils import secure_filename

app=Flask(__name__)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights("model_weights.h5")

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    
def video():
    cap=cv2.VideoCapture(0)
    a=0
    b=0
    d=[]

    while 1:
        ret,frame=cap.read()

        if not ret:
            break

        g_i=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        faces_detected = face_haar_cascade.detectMultiScale(g_i, 1.32, 5)
        for (x,y,w,h) in faces_detected:  
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),thickness=3)  
                roi_gray=g_i[y:y+w,x:x+h]
                roi_gray=cv2.resize(roi_gray,(48,48))  
                img = roi_gray.reshape((1,48,48,1))
                img = img /255.0

                max_index = np.argmax(model.predict(img.reshape((1,48,48,1))), axis=-1)[0]

                global predicted 
                predicted = emotions[max_index] 
                d.append(predicted)

                if predicted:
                    a += 1
                    print(a)

                cv2.putText(frame, predicted, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                if a>=20:
                    predicted="""You are in {} """.format(predicted)
                    b +=1
                    print(b)
                    cv2.putText(frame, predicted, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (1,1,1), 2)
        resized_img = cv2.resize(frame, (1000, 700))  
        ret, buffer = cv2.imencode('.jpg', resized_img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        if b==15:
            cap.release()  
            cv2.destroyAllWindows()
            
            break

def image(files):
    global predicted_emotion
    a=[]
    
    c_img = cv2.imread(files)
    gray_img = cv2.cvtColor(c_img, cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
    for (x,y,w,h) in faces_detected:  
        cv2.rectangle(c_img,(x,y),(x+w,y+h),(255,0,0),thickness=3)  
        roi_gray=gray_img[y:y+w,x:x+h]
        roi_gray=cv2.resize(roi_gray,(48,48))  
        img = roi_gray.reshape((1,48,48,1))
        img = img /255.0

        max_index = np.argmax(model.predict(img.reshape((1,48,48,1))), axis=-1)[0]

                  
        predicted_emotion = emotions[max_index]  
        print("Expression Statement : ",predicted_emotion)

        cv2.putText(c_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    resized_img = cv2.resize(c_img, (1000, 700))  
    # cv2.imshow('Facial emotion analysis ',resized_img)
    cv2.imwrite("static/output.jpg", resized_img)
    predicted_emotion=str(predicted_emotion)
    a.append(predicted_emotion)
    print("last : ",a)
    if cv2.waitKey(0) == ord('s'):
        cv2.destroyAllWindows()
    

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video')
def video1():
    return render_template('livevideo.html')

@app.route('/video-feed')
def video_feed():
    return Response(video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/image')
def image1():
    global a
    return render_template('image.html')

@app.route('/pic_image',methods=['POST'])
def image2():
    global predicted_emotion
    file=request.files['file']
    files=file.filename
    print(files)   
    basepath = os.path.dirname(__file__)
    print(basepath)
    file_path = os.path.join(basepath, 'upload',secure_filename(file.filename))
    file.save(file_path)   
    image(file_path)
    return render_template('upload.html',p=predicted_emotion)

app.run(debug=True)   