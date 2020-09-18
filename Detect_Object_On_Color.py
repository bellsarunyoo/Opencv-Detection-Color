import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0)
cv2.namedWindow("Trackbars")

cv2.createTrackbar("L – H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L – S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L – V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U – H", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U – S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U – V", "Trackbars", 255, 255, nothing)
# cv2.createTrackbar("U – H", "Trackbars", 0, 179, nothing)
# cv2.createTrackbar("U – S", "Trackbars", 0, 255, nothing)
# cv2.createTrackbar("U – V", "Trackbars", 0, 255, nothing)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray_blur=cv2.GaussianBlur(hsv,(25,25),4)#Delete Nois
    
#     l_h = cv2.getTrackbarPos("L – H", "Trackbars")
#     l_s = cv2.getTrackbarPos("L – S", "Trackbars")
#     l_v = cv2.getTrackbarPos("L – V", "Trackbars")
#     u_h = cv2.getTrackbarPos("U – H", "Trackbars")
#     u_s = cv2.getTrackbarPos("U – S", "Trackbars")
#     u_v = cv2.getTrackbarPos("U – V", "Trackbars")
    l_h = 135
    l_s = 34
    l_v = 194
    
    u_h = 255
    u_s = 255
    u_v = 255
     
    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(gray_blur, lower_blue, upper_blue)

    result = cv2.bitwise_and(frame, frame, mask=mask)

    counter=0 
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for con in contours:
        x,y,w,h = cv2.boundingRect(con)
        if x != 0 | y != 0:
            counter += 1
            cv2.rectangle(frame,(x,y),(x+w,y+h),(100,255,0),2)
            cv2.rectangle(result,(x,y),(x+w,y+h),(100,255,0),2)

       
    cv2.putText(frame,'Detect : '+ str(counter),(10,100), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2,cv2.LINE_AA) 
    cv2.imshow("original", hsv)
    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("result", result)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()