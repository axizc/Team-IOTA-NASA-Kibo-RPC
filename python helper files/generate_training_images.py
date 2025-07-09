import cv2
import numpy as np
import os
import random
li= ["coin","compass","coral","crystal","diamond","emerald","fossil","key","letter","shell","treasure_box"]

def load_pngs(folder):
    images = {"coin":[],"compass":[],"coral":[],"crystal":[],"letter":[], "diamond":[], "emerald":[], "fossil":[], "key":[], "shell":[], "treasure_box":[]}
    li= ["coin","compass","coral","crystal","diamond","emerald","fossil","key","letter","shell","treasure_box"]
    for a in li:
        for filename in os.listdir(folder+a):
            if filename.lower().endswith('.png'):
                print(filename, a)
                img = cv2.imread(os.path.join(folder+a, filename), cv2.IMREAD_UNCHANGED)
                if img is not None and img.shape[2] == 4:
                    images[a].append(img)
    return images

def random_resize(img, min_size=30, max_size=100):
    h, w = img.shape[:2]
    scale = random.uniform(min_size / max(h, w), max_size / max(h, w))
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized

def paste_image(bg, overlay, x, y):
    h, w = overlay.shape[:2]
    alpha = overlay[:, :, 3] / 255.0
    for c in range(3):
        bg[y:y+h, x:x+w, c] = (1 - alpha) * bg[y:y+h, x:x+w, c] + alpha * overlay[:, :, c]

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0
    return inter_area / union_area

# Allow small overlap
def is_too_much_overlap(x, y, w, h, placed, max_iou=0.15):
    for (px, py, pw, ph) in placed:
        if compute_iou((x, y, w, h), (px, py, pw, ph)) > max_iou:
            return True
    return False

# # Canvas
bg_width, bg_height = 2970, 2100
background = np.ones((bg_height, bg_width, 3), dtype=np.uint8) * 255
object_images = load_pngs("C:\\Users\\vt\\Downloads\\kibo\\rotated_nobg\\")
placed_boxes = []

# # Place objects
for iy in range(5000):
    background = np.ones((bg_height, bg_width, 3), dtype=np.uint8) * 255
    placed_boxes = []
    for _ in range(10):
        i= random.randint(0, 10)
        img = random.choice(object_images[li[i]])
        img_resized = random_resize(img, min_size=100, max_size=800)
        h, w = img_resized.shape[:2]
        
        for _ in range(1000):
            x = random.randint(0+w, bg_width - w)
            y = random.randint(0+h, bg_height - h)

            if not is_too_much_overlap(x, y, w, h, placed_boxes, max_iou=0.16):  # Allow small overlap
                paste_image(background, img_resized, x, y)
                placed_boxes.append((x, y, w, h))
                with open("C:\\Users\\vt\\Downloads\\kibo\\txt\\"+str(iy+35000)+".txt", "a") as file:
                    file.write(str(i)+" "+str((x+w/2)/bg_width)+" "+str((y+h/2)/bg_height)+" "+str(w)+" "+str(h)+" "+"\n")

                break

    # # Show result
        cv2.imwrite("C:\\Users\\vt\\Downloads\\kibo\\mixed\\"+str(iy+35000)+".png", background)
