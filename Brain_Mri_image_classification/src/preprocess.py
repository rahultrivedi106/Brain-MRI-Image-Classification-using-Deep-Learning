import os, cv2
from pathlib import Path

def resize_and_equalize(src_dir, dst_dir, size=(224,224)):
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif')):
                src = os.path.join(root, f)
                img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, size)
                img = cv2.equalizeHist(img)
                dst = os.path.join(dst_dir, f)
                cv2.imwrite(dst, img)

if __name__ == '__main__':
    resize_and_equalize('data/raw/train','data/processed/train')
