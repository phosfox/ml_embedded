import glob
import sys
import cv2

def main():
    path = "C:\\Users\\Constantin\\Downloads\\og"
    target_path = "C:\\Users\\Constantin\\Downloads\\relabeled"
    img_paths = [p for p in glob.glob(f"{path}\\*\\*.jpg")]
    for idx, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)
        cv2.imshow("img", img)
        key = cv2.waitKey(0)
        if key == ord('w'):
            straight_path = f"{target_path}\\straight\\{idx}.jpg"
            print(straight_path)
            cv2.imwrite(straight_path, img)
        elif key == ord("a"):
            left_path = f"{target_path}\\left\\{idx}.jpg"
            print(left_path)
            cv2.imwrite(left_path, img)
        elif key == ord("d"):
            right_path = f"{target_path}\\right\\{idx}.jpg"
            print(right_path)
            cv2.imwrite(right_path, img)
        else:
            print("else")
            break

if __name__ == "__main__":
    main()
