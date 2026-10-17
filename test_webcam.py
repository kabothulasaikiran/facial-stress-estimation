# test_webcam.py
# Simple test to check if webcam works

import cv2

print("Testing webcam...")

# Try camera index 0, 1, and 2
for i in range(3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"  Camera index {i} — WORKS!")
        ret, frame = cap.read()
        if ret:
            print(f"  Frame captured successfully!")
            print(f"  Frame size: {frame.shape}")
            cv2.imshow(f"Camera {i} Test", frame)
            cv2.waitKey(3000)  # Show for 3 seconds
            cv2.destroyAllWindows()
        cap.release()
    else:
        print(f"  Camera index {i} — NOT found")

print("\nDone. If no camera worked check steps below.")