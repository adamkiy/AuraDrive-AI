from sensor import EyeBlinkSensor
import cv2

sensor = EyeBlinkSensor(debug=True)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    out = sensor.process_frame(frame)

    cv2.imshow("AuraDrive - Driver Fatigue Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()