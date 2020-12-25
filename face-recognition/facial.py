import face_recognition
import cv2

known = './img/known/Barack Obama.jpg'
image_of_bill = face_recognition.load_image_file(known)
person = known.split('/')[-1].split('.')[0]
bill_face_encoding = face_recognition.face_encodings(image_of_bill)[0]

unknown = './img/unknown/barack-obama-12782369-1-402.jpg'

unknown_image = face_recognition.load_image_file(unknown)
unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

# Facial Recognition
results = face_recognition.compare_faces(
    [bill_face_encoding], unknown_face_encoding)

if results[0]:
    print('This is {}'.format(person))
else:
    print('This is NOT {}'.format(person))

# Facial Features

image = face_recognition.load_image_file(unknown)
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

face_landmarks_list = face_recognition.face_landmarks(image)
print(face_landmarks_list,"\n\n")

for feature in face_landmarks_list[0]:
    #print(feature)
    for x in face_landmarks_list[0][feature]:
        cv2.circle(image,x, 1, (0,255,0), 2)

cv2.imshow('Output',image)
cv2.waitKey(0)

#For changing the threshold for sensitivity of face recognition
#ap = argparse.ArgumentParser()
#ap.add_argument("-c", "--confidence", type=float, default=0.5,
#	help="minimum probability to filter weak detections")