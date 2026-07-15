import cv2
from picamera2 import Picamera2
import time
from gpiozero import Buzzer

# ---------------- BUZZER SETUP ----------------
buzzer = Buzzer(18) # GPIO pin for buzzer (BCM)
# ------------------------------------------------

# Load MobileNet SSD model
net = cv2.dnn.readNetFromCaffe(
"deploy.prototxt",
"mobilenet_iter_73000.caffemodel"

)

# Animals & birds we care about
animal_classes = ["dog", "cat", "horse", "sheep", "cow", "bird"]

# MobileNet SSD classes
CLASSES = [
"background", "aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair", "cow",
"diningtable", "dog", "horse", "motorbike", "person",
"pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Initialize camera
picam = Picamera2()
picam.configure(picam.create_preview_configuration(main={"size": (640, 480)}))
picam.start()
time.sleep(2)

print(" Starting Animal Detection... Press 'q' to quit")

# Store last detected animals (to avoid spam)
last_detected = set()

while True:
frame = picam.capture_array()

# Convert BGRA → BGR if needed
if frame.shape[2] == 4:

frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

# Prepare image for detection
blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
net.setInput(blob)
detections = net.forward()

detected_animals = []

# Detection loop
for i in range(detections.shape[2]):
confidence = detections[0, 0, i, 2]

if confidence > 0.5:
class_id = int(detections[0, 0, i, 1])
label = CLASSES[class_id]

if label in animal_classes:
detected_animals.append(label)

# Bounding box
box = detections[0, 0, i, 3:7] * [
frame.shape[1], frame.shape[0],
frame.shape[1], frame.shape[0]
]
(x1, y1, x2, y2) = box.astype("int")

# Draw rectangle
cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Show BIG label
cv2.putText(frame, label.upper(), (x1, y1 - 10),
cv2.FONT_HERSHEY_SIMPLEX, 1,
(0, 0, 255), 3)

# ---------------- BUZZER + PRINT LOGIC ----------------
if detected_animals:
unique_animals = set(detected_animals)

# Print only when new animal appears
if unique_animals != last_detected:
print(" Animal Detected:", ", ".join(unique_animals))
last_detected = unique_animals

# Buzzer beep
buzzer.on()
time.sleep(0.3)
buzzer.off()

else:
last_detected = set()
buzzer.off()

# Show camera
cv2.imshow("Animal Detection", frame)

# Exit
if cv2.waitKey(1) & 0xFF == ord('q'):

break

# Cleanup
buzzer.off()
cv2.destroyAllWindows()
