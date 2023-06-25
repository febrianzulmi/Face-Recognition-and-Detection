import cv2
import numpy as np
import os
import json
import paho.mqtt.client as mqtt
import time

# MQTT broker configuration
mqtt_broker = "159.223.61.133"
mqtt_port = 1883
mqtt_topic = "door/status"
mqtt_username = "mecharoot"
mqtt_password = "mecharnd595"

# set up the thermal camera index (thermal_camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) on Windows OS)
thermal_camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# set up the thermal camera resolution
thermal_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
thermal_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# load the haar cascade face detector
haar_cascade_face = cv2.CascadeClassifier(
    "C:/Users/Febrian Zulmi/Downloads/haar-cascade-files-master/haar-cascade-files-master/haarcascade_frontalface_alt2.xml")
# fever temperature threshold in Celsius or Fahrenheit
fever_temperature_threshold = 70.0

# create directory to save captured images
save_directory = "captured_images"
os.makedirs(save_directory, exist_ok=True)

# counter for captured images
image_counter = 0

# initialize MQTT client
mqtt_client = mqtt.Client()

# set username and password for MQTT broker
mqtt_client.username_pw_set(mqtt_username, mqtt_password)

# connect to MQTT broker
mqtt_client.connect(mqtt_broker, mqtt_port, 60)

# MQTT callback function for when a message is received


def on_message(client, userdata, msg):
    payload = msg.payload.decode("utf-8")
    print("Received message:", payload)

    # parse the JSON payload
    data = json.loads(payload)
    door_status = data["door_status"]

    # process the door status
    if door_status == "1":
        print("Pintu Terbuka: Suhu anda aman")
    elif door_status == "0":
        print("Pintu Tertutup: Suhu anda terlalu tinggi")
    else:
        print("Invalid door status")


# set the callback function
mqtt_client.on_message = on_message

# subscribe to MQTT topic
mqtt_client.subscribe(mqtt_topic)

# start the MQTT loop to handle incoming messages
mqtt_client.loop_start()

# initialize last_message_time
last_message_time = time.time()

# flag to track first reading
first_reading = True

# loop over the thermal camera frames
while True:
    # grab the frame from the thermal camera stream
    (grabbed, frame) = thermal_camera.read()

    # convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the input frame using the haar cascade face detector
    faces = haar_cascade_face.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # variable to store door status (1: door open, 0: door closed)
    door_status = 0

    # loop over the bounding boxes to measure their temperature
    for (x, y, w, h) in faces:
        # draw the rectangles
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)

        # extract the region of interest (face) from the frame
        face_roi = frame[y:y + h, x:x + w]

        # calculate the average temperature based on the pixel values in the face_roi
        temperature = np.mean(face_roi)

        # write temperature value on the frame
        if temperature > fever_temperature_threshold:
            # red text: fever temperature
            cv2.putText(frame, "{0:.1f} Celsius".format(temperature), (x, y - 10), cv2.FONT_HERSHEY_PLAIN,
                        1, (0, 0, 255), 2)
        else:
            # white text: normal temperature
            cv2.putText(frame, "{0:.1f} Celsius".format(temperature), (x, y - 10), cv2.FONT_HERSHEY_PLAIN,
                        1, (255, 255, 255), 2)
            # set door status to open
            door_status = 1

        # save the captured face image
        if temperature > fever_temperature_threshold and image_counter < 5:
            image_path = os.path.join(
                save_directory, f"captured_image_{image_counter}.jpg")
            cv2.imwrite(image_path, face_roi)
            print(f"Captured image {image_counter+1} saved.")
            image_counter += 1

    # check if it's time to send the MQTT message
    if (time.time() - last_message_time >= 10) or first_reading:
        # create JSON payload
        payload = json.dumps({"door_status": str(door_status)})

        # publish door status to MQTT broker
        mqtt_client.publish(mqtt_topic, payload)

        # update last_message_time
        last_message_time = time.time()

        # set first_reading flag to False
        first_reading = False

    # display the frame
    cv2.imshow("Face Temperature", frame)

    # check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the thermal camera, close all windows, and disconnect from MQTT broker
thermal_camera.release()
cv2.destroyAllWindows()
mqtt_client.loop_stop()
mqtt_client.disconnect()
