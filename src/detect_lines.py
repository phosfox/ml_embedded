import cv2  # OpenCV - Bildverarbeitungsbibliothek
import numpy as np  # Numpy - Bibliothek für Matrizen, Arrays etc.
from pathlib import Path

def detect_lines(img):
    x, y, _ = img.shape
    pts = np.array(
        [[[y/3, x/2], [y/3*2, x/2], [y, x], [0, x]]], dtype=np.int32)
    black_img = np.zeros_like(img)
    roi_image = cv2.fillPoly(black_img, pts, (255, 255, 255))
    cropped_img = np.bitwise_and(img, roi_image)

    gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

    low_white = np.array([235])
    up_white = np.array([255])
    mask_img = cv2.inRange(gray_img, low_white, up_white)

    edges_img = cv2.Canny(mask_img, 75, 150)

    lines_arr = cv2.HoughLinesP(edges_img, 1, np.pi/180, 50, minLineLength=5, maxLineGap=100)

    if lines_arr is not None:
        for line in lines_arr:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    bounding_rect = cv2.polylines(img, pts, True, (255,0,0), 3)
    return img

if __name__ == "__main__":
    video_path = ".\\videos\\car_pov.mp4"
    print(video_path)
    video = cv2.VideoCapture(video_path)

    while True:
        ret, frame = video.read()
        print(frame.shape)
        if not ret:
            video = cv2.VideoCapture(video_path)
            continue

        frame = detect_lines(frame)
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()
