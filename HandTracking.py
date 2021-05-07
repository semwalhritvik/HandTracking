# Importing packages
import cv2
import mediapipe as mp
import time

# Using the default webcam(0) for videocapture
cap = cv2.VideoCapture(0)

pTime=0 
cTime=0 

# The hand function by mediapipe contains predefined functions for recognizing hands 
mpHands = mp.solutions.hands
hands = mpHands.Hands() 

# For plotting the landmarks on the image
mpDraw = mp.solutions.drawing_utils 


while True:
    # Processing the video as individual images
    success, img = cap.read()    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB) 
    #print(result.multi_hand_landmarks)

    # To deal with multiple hands in the scene
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            #p rinting the id of each hand landmark
            for id,lm in enumerate(handLms.landmark): 
                #print(id,lm)
                h,w,c = img.shape # height, width, channel
                cx, cy = int(lm.x * w), int(lm.y * h) 
                
                
                # Color one specific landmark
                # For example 8 is the id for tip of index finger
                if id==8:
                   cv2.circle(img, (cx,cy), 15, (255,0,0),cv2.FILLED) 
                   print(id,cx,cy) # printing id along with the centres

            # Plotting the points of landmarks and joining then with connections    
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    # Getting frames per second (FPS)
    cTime = time.time()  
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (5,40), cv2.FONT_HERSHEY_PLAIN, 2, (100,255,0),3)

    cv2.imshow("Image", img)
    
    # Wait for ESC key to exit
    k = cv2.waitKey(1) & 0xFF    
    if k == 27: 
        cv2.destroyAllWindows()
        break
    