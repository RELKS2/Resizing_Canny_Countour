import cv2 as cv

faceCascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

img1 = cv.imread('images/img2.jpg')
#reads images
img = cv.imread('images/img1.jpg')

#display image in new window
cv.imshow('cat', img)

# 0 builtin webcam
capture = cv.VideoCapture(0)

#makes image grey
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

#shows the outlines of the image(black and white) 
canny = cv.Canny(img, 125, 180)
cv.imshow('Canny Edges', canny)

#resize image

def rescaleFrame(frame, scale=0.75):
    width= int(frame.shape[1]* scale) 
    height=int(frame.shape[0]* scale) 
    dimensions = (width,height)

    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

def changeRes(width,height):
    #live video 
    capture.set(3,width)
    capture.set(4,height)

while True:
    isTrue, frame = capture.read()
    frame_resized =rescaleFrame(frame)
    cv.imshow('Video', frame)
    cv.imshow('video Resized', frame_resized) 

    isTrue, frame = capture.read()
    
    cv.imshow('Video', frame)
    imgGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(imgGray, 1.3, 5)
    
    
    #rectangle around face
    for (x, y, w, h) in faces:
        frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv.imshow('Detected Faces', frame)


    if cv.waitKey(20) & 0xFF== ord('d'):
        break

capture.release()
cv.destroyAllWindows()