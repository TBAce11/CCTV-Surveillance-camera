import cv2, time, datetime, os, config
from twilio.rest import Client

# replace these lines with your Twilio credentials
account_sid = os.environ.get("TWILIO_ACCOUNT_SID", config.TWILIO_ACCOUNT_SID)
auth_token = os.environ.get("TWILIO_AUTH_TOKEN", config.TWILIO_AUTH_TOKEN)
phone_from = os.environ.get("TWILIO_PHONE_FROM", config.TWILIO_PHONE_FROM)
phone_to = os.environ.get("TWILIO_PHONE_TO", config.TWILIO_PHONE_TO)

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     "haarcascade_fullbody.xml")
detection = False
detection_stopped_time = None
timer_started = False
SECONDS_TO_RECORD_AFTER_DETECTION = 5

frame_size = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

while True:
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # gray scale image
    faces = face_cascade.detectMultiScale(gray, 1.3, 3)  # returns a list of positions of all the faces that exist
    bodies = face_cascade.detectMultiScale(gray, 1.3, 3)

    # face or body detected
    if len(faces) + len(bodies) > 0:
        # recording already in progress
        if detection:
            timer_started = False
        # no previous recording so signal for new video to be taken sent
        else:
            detection = True
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out = cv2.VideoWriter(f"{current_time}.mp4", fourcc, 20, frame_size)
            print("Recording in progress.")

            # insert Twilio code to crop image and send message with it here

            client = Client(account_sid, auth_token)
            client.api.account.messages.create(
                body="You've got company! Go check your files for the recording.",
                from_=phone_from,
                to=phone_to,
                #media_url=url
            )

    # no face or body detected right now, but were a few seconds ago
    elif detection:
        if timer_started:
            # 5 seconds delay of inactivity reached
            if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                detection = False
                timer_started = False
                out.release()
                print("Recording stopped.")
        else:
            timer_started = True
            detection_stopped_time = time.time()

    if detection:
        out.write(frame)

    cv2.imshow("Camera", frame)

    # If q key is pressed, stop the program
    if cv2.waitKey(1) == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()
