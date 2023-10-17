# Pool recognition

## Install dependencies 
Create a python venv:  
``` python -m venv ./  ``` 

Install required dependencies :
``` ./bin/pip3 install face_recognition numpy requests ``` 

And for RaspberryPi :
``` ./bin/pip3 install picamera```

Or for standard webcam :
``` ./bin/pip3 install opencv-contrib-python```

And clone the code :
``` git clone git@github.com:DonNav/Pool-recognition.git```

## OpenCV or RaspberryPi

## Add faces
In order to reconize people, add pictures in a folder names "./images"
Then, in pool-recoginiton.py modify line 31 and 32
```commandline
myface_image = face_recognition.load_image_file("./images/myface.jpg")
myface_face_encoding = face_recognition.face_encodings(myface_image)[0]
```
Repeat those lignes for all people you want to reconize.
And then, modify line 63 in order to display the name :
```
        if match[0]:
            name = "My first face"
        else if match[1]:
            name = "My second face"
...
```

## Use it
To launch the program, use :
```commandline
./bin/python3 ./pool_recognition.py
```