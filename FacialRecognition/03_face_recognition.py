import cv2


video = cv2.VideoCapture(0)

facedetect = cv2.CascadeClassifier(
    "C:/Users/Febrian Zulmi/Downloads/haar-cascade-files-master/haar-cascade-files-master/haarcascade_frontalface_alt2.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(
    "D:/TUGAS PKL/facerecognition/pengolahan citra untuk suhu/trainer/trainer.yml")

name_list = ["", "Zulmi", "Pravas"]

imgBackground = cv2.imread(
    "C:/Users/Febrian Zulmi/Pictures/WhatsApp Image 2022-11-13 at 20.44.55.jpeg")

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        serial, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        if confidence < 100:
            confidence = int(100 * (1 - confidence/400))
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
            cv2.putText(frame, name_list[serial], (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f" {confidence}%", (x+100, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
            cv2.putText(frame, "Unknown", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    frame = cv2.resize(frame, (640, 480))
    imgBackground[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow("Face Recognition", imgBackground)

    k = cv2.waitKey(1)
    if k == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
