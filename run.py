from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2

(major, minor) = cv2.__version__.split(".")[:2]

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
go = True

# for cv2 version lower than 3.2
#tracker = cv2.Tracker_create(args["tracker"].upper())
tracker = cv2.TrackerKCF_create()

initBB = None

print("Web Cam starting...")
vs = VideoStream(src=0).start()
time.sleep(1.0)


while True:

	frame = vs.read()
	
	if frame is None:
		break

	frame = imutils.resize(frame, width=500)
	(H, W) = frame.shape[:2]

	if initBB is not None:
		(success, box) = tracker.update(frame)

		if success:
			(x, y, w, h) = [int(v) for v in box]
			cv2.rectangle(frame, (x, y), (x + w, y + h),
				(0, 255, 0), 2)
		else:	
			go = True
			initBB = None
			print("Face Lost")
			tracker.clear()
		
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	img = frame
	if go == True:
		gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		coords = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
		biggest = coords
		if len(coords) > 1:
			biggest = (0, 0, 0, 0)
			for i in coords:
				if i[3] > biggest[3]:
					biggest = i
			biggest = np.array([i], np.int32)
		elif len(coords) == 1:
			biggest = coords

		for (x, y, w, h) in biggest:
			print("Face Found")
			cv2.rectangle(frame, (x, y), (x + w, y + h),
				(0, 255, 0), 2)
			frame = img[y:y + h, x:x + w]
			initBB = (x,y,w,h)
			tracker.init(frame, initBB)
			go = False

	if key == ord("q"):
		break

vs.stop()
cv2.destroyAllWindows()
