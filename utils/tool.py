import numpy as np
import cv2
import datetime
import os
import json

##########################################################################
def get_poly_box_wh(poly_box): # (4,2)
    lt, rt, rb, lb = poly_box
    w = int((np.linalg.norm(lt - rt) + np.linalg.norm(lb - rb)) // 2)
    h = int((np.linalg.norm(lt - lb) + np.linalg.norm(rt - rb)) // 2)
    return w, h


def crop_obj_in_bg(bg_img, poly, w, h):
    poly = poly.astype(np.float32)
    pos = np.float32([[0,0], [w,0], [w,h], [0,h]])
    M = cv2.getPerspectiveTransform(poly, pos)
    obj_img = cv2.warpPerspective(bg_img, M, (w, h))
    return obj_img, M

##########################################################################
def get_time_str(human_mode=False):
    now = datetime.datetime.now()
    if human_mode:
        s = f"{now.year:04d}-{now.month:02d}-{now.day:02d} {now.hour:02d}:{now.minute:02d}:{now.second:02d}"
    else:
        s = f"{now.year:04d}{now.month:02d}{now.day:02d}{now.hour:02d}{now.minute:02d}{now.second:02d}"
        s += f"_{now.microsecond:06d}"
    return s

##########################################################################
def fix_ratio_resize_img(img, size, target='w'):
    h, w = img.shape[:2]
    ratio = h/w
    if target == 'w': resized_img = cv2.resize(img, dsize=(size, int(ratio * size)))
    else:             resized_img = cv2.resize(img, dsize=(int(size / ratio), size))
    return resized_img

##########################################################################
def clear_Q(Q):
    with Q.mutex:
        Q.queue.clear()
        
def clear_serial(ser):
    while True:
        if ser.read_all() == b'': break
        
######################################################################
def draw_box_text(img, data, poly_boxes):
    img = img.copy()
    for text, poly in zip(data, poly_boxes):
        cv2.polylines(img, [poly], True, (255,0,0))
        text_loc = np.min(poly, axis=0)
        text_loc[1] -= 10
        cv2.putText(img, text, text_loc, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    return img

##########################################################################
def manage_file_num(dir_path, max_size=500, num_remove=100):
    path = os.path.join(dir_path, "*.jpg")
    img_paths = sorted(path)
    if len(img_paths) < max_size: return

    for path in img_paths[:num_remove]:
        os.remove(path)
        
##########################################################################
def imread(path, mode=cv2.IMREAD_COLOR):
    encoded_img = np.fromfile(path, np.uint8)
    img = cv2.imdecode(encoded_img, mode)
    return img

def imwrite(path, img):
    result, encoded_img = cv2.imencode('.jpg', img)
    if result:
        with open(path, 'w') as f:
            encoded_img.tofile(f)
    return result

##########################################################################