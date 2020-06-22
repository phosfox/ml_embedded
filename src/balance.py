import os
import shutil
import glob
import random
def main():
    src = "C:\\Users\\Constantin\\Downloads\\relabeled\\straight_full\\"
    tar = "C:\\Users\\Constantin\\Downloads\\relabeled\\straight\\"
    paths = [p for p in glob.glob(f"{src}*.jpg")]
    random.shuffle(paths)
    for p in paths[:200]:
        filename = p.split('\\')[-1]
        shutil.copy(p, f"{tar}{filename}") 
if __name__ == "__main__":
    main()
