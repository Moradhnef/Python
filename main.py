
import cv2
import face_recognition
from simple_facerec import SimpleFacerec

# Load Camera
cap = cv2.VideoCapture(0) #(0) zero means load the first camera

#Encode faces from a folder
sfr=SimpleFacerec()
sfr.load_encoding_images("images/") #path for all the images

while True: #to get frame after frame
    ret, frame = cap.read()

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)   #we will get the face rectangle for ea face and then the face name
    for face_loc, name in zip(face_locations, face_names): #we get the location of the face so we need to
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3] #y1 as top left bottom right

        #customize the frame+name
        cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2) #to add name
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4) #rectange frame with color and thickness

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27: #27 is the ESC in the keyboard
        break

cap.release()
cv2.destroyAllWindows()



#when we see one of the faces from the images then we get a name and if we get a new face then we get unknown.

#testing single images

#img = cv2.imread("Messi1.webp")
#rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#img_encoding = face_recognition.face_encodings(rgb_img)[0]

#img2 = cv2.imread("images/Elon Musk.jpg")
#rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
#img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

#compare images faces and return true/false
#result = face_recognition.compare_faces([img_encoding], img_encoding2)
#print("Result: ", result)

#cv2.imshow("Img",img)
#cv2.imshow("Img 2",img2)
#cv2.waitKey(0)