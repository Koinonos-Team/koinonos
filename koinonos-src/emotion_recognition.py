import time
from io import BytesIO
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import FaceAttributeType
import cv2


KEY = "ea420711a8d34667b546da5a5f3c1d00"
ENDPOINT = "https://koinonos-face.cognitiveservices.azure.com/"

# Create an authenticated FaceClient.
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

# Convert width height to a point in a rectangle
def getRectangle(faceDictionary):
    rect = faceDictionary.face_rectangle
    left = rect.left
    top = rect.top
    right = left + rect.width
    bottom = top + rect.height
    
    return ((left, top), (right, bottom))



cap = cv2.VideoCapture(0)
# Black color in BGR 
color = (0, 0, 0) 
# Thickness of -1 will fill the entire shape 
thickness = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Detect faces
    buf = cv2.imencode('.jpg', frame) [1]
    detected_faces = face_client.face.detect_with_stream(image=BytesIO(buf), detectionModel='detection_02', return_face_attributes=[FaceAttributeType.emotion])


    # Draw rectangle on webcam
    for face in detected_faces:
        start, end = getRectangle(face)
        frame = cv2.rectangle(frame, start, end, color, thickness) 

        # Display emotions
        loc = (50, 50)
        for emotion, score in face.face_attributes.emotion.as_dict().items():
            cv2.putText(frame, f'{emotion}: {score}', loc, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
            loc = (loc[0], loc[1] + 30)
    
        

    # Display the resulting frame
    cv2.imshow('frame',frame)
    time.sleep(2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()