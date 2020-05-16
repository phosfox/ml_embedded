import cv2  # OpenCV - Bildverarbeitungsbibliothek
import numpy as np  # Numpy - Bibliothek für Matrizen, Arrays etc.
from pathlib import Path
import logging
import math


def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    if slope == 0:  # prevents div by 0
        slope = 0.001
    y1 = height  # bottom of the frame
    y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]


def average_slope_intercept(frame, line_segments):
    """
    This function combines line segments into one or two lane lines
    If all line slopes are < 0: then we only have detected left lane
    If all line slopes are > 0: then we only have detected right lane
    """
    lane_lines = []
    if line_segments is None:
        logging.info('No line_segment segments detected')
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1/3
    # left lane line segment should be on left 2/3 of the screen
    left_region_boundary = width * (1 - boundary)
    # right lane line segment should be on left 2/3 of the screen
    right_region_boundary = width * boundary

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                logging.info(
                    'skipping vertical line segment (slope=inf): %s' % line_segment)
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope, intercept = fit
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    # [[[316, 720, 484, 432]], [[1009, 720, 718, 432]]]
    logging.debug('lane lines: %s' % lane_lines)

    return lane_lines


def detect_lines(img):
    x, y, _ = img.shape
    pts = np.array(
        [[[y/2, x/2], [y, x], [0, x]]], dtype=np.int32)
    black_img = np.zeros_like(img)
    roi_image = cv2.fillPoly(black_img, pts, (255, 255, 255))
    cropped_img = np.bitwise_and(img, roi_image)

    gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

    low_white = np.array([235])
    up_white = np.array([255])
    mask_img = cv2.inRange(gray_img, low_white, up_white)

    edges_img = cv2.Canny(mask_img, 75, 150)

    lines_arr = cv2.HoughLinesP(
        edges_img, 1, np.pi/180, 50, minLineLength=50, maxLineGap=100)
    lane_lines = average_slope_intercept(img, lines_arr)

    return lane_lines


def steering_angle_helper(x_offset, y_offset):
        # angle (in radian) to center vertical line
    print("offset: ", x_offset, y_offset)
    angle_to_mid_radian = math.atan(x_offset / y_offset)
    # angle (in degrees) to center vertical line
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
    # this is the steering angle needed by picar front wheel
    return angle_to_mid_deg + 90


def calc_steering_angle(frame, lane_lines):
    height, width, _ = frame.shape
    if not lane_lines:
        return 90
    if len(lane_lines) < 2:
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
        y_offset = int(height / 2)
        return 90
        #return steering_angle_helper(x_offset, y_offset)

    if len(lane_lines) == 2:
        left, right = lane_lines
        print("left:", left)
        print("right:", right)
        _, _, left_x2, _ = lane_lines[0][0]
        _, _, right_x2, _ = lane_lines[1][0]
        mid = int(width / 2)
        x_offset = (left_x2 + right_x2) / 2 - mid
        y_offset = int(height / 2)
        return steering_angle_helper(x_offset, y_offset)


def calc_heading_line(frame, steering_angle):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    # figure out the heading line from steering angle
    # heading line (x1,y1) is always center bottom of the screen
    # (x2, y2) requires a bit of trigonometry

    # Note: the steering angle of:
    # 0-89 degree: turn left
    # 90 degree: going straight
    # 91-180 degree: turn right
    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    return (x1, y1, x2, y2)


prev_angle = 90


def smooth_angle(new_angle, tolerance):
    global prev_angle
    if new_angle >= prev_angle - tolerance and new_angle <= prev_angle + tolerance:
        prev_angle = new_angle
    if new_angle <= prev_angle:
        new_angle = prev_angle - tolerance
    if new_angle >= prev_angle:
        new_angle = prev_angle + tolerance
    else:
        new_angle = prev_angle
    return new_angle


if __name__ == "__main__":
    video_path = ".\\videos\\car_pov.mp4"
    video = cv2.VideoCapture(video_path)
    while video.isOpened():
        ret, frame = video.read()
        height, width, _ = frame.shape
        lines = detect_lines(frame)
        steering_angle = calc_steering_angle(frame, lines)
        steering_angle = smooth_angle(steering_angle, 5)
        print("Angle:", steering_angle)
        heading_line = calc_heading_line(frame, steering_angle)
        print("Heading_line:", heading_line)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                h_x1, h_y1, h_x2, h_y2 = heading_line
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.line(frame, (h_x1, h_y1), (h_x2, h_y2), (0, 0, 255), 3)
        cv2.putText(frame, str(steering_angle), (int(width/2), height),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()
