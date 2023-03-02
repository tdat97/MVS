from utils.logger import logger
from utils.text import *
from utils import tool, code
from PIL import ImageFont, ImageDraw, Image, ImageTk
from threading import Thread, Lock
import numpy as np
import time
import cv2
import os

#######################################################################
def recode(self):
    dir_dic = {'raw':SAVE_RAW_IMG_DIR, 'debug':SAVE_DEBUG_IMG_DIR, None:SAVE_NG_IMG_DIR}
    for code in self.code2name:
        dir_dic[code] = os.path.join(SAVE_OK_IMG_DIR, code)
        if not os.path.isdir(dir_dic[code]): os.mkdir(dir_dic[code])
    
    while not self.stop_signal:
        time.sleep(0.01)
        
        if self.recode_Q.empty(): continue
        img, name = self.recode_Q.get()
        file_name = f"{tool.get_time_str()}.jpg"
        
        if img is None: continue
        
        # 1ch -> 3ch
        if len(img.shape) == 2:
            img = cv2.merge([img,img,img])
        
        # 코드명 폴더 만들기
        if not name in dir_dic:
            path = os.path.join(SAVE_OK_IMG_DIR, name)
            if not os.path.isdir(path): os.mkdir(path)
            dir_dic[name] = path
            
        # save image
        path = os.path.join(dir_dic[name], file_name)       
        tool.imwrite(path, img)
        tool.manage_file_num(dir_dic[name])

#######################################################################
def snap(self):
    self.thr_lock.acquire()
    self.serial.write(BYTES_DIC["light_on"])
    self.thr_lock.release()
    
    time.sleep(0.07)
    # self.cam.set_exposure(2500)
    img = self.cam.get_image()
    
    self.thr_lock.acquire()
    self.serial.write(BYTES_DIC["light_off"])
    self.thr_lock.release()
    
    self.raw_Q.put(img)
    
#######################################################################
def snaper(self):
    sensor_lock = False
    
    while not self.stop_signal:
        time.sleep(0.05)
        
        self.thr_lock.acquire()
        self.serial.write(BYTES_DIC["get_sensor1"])
        value = self.serial.read(4)
        self.thr_lock.release()
        
        if not sensor_lock and value[0] != 0xff and value[2] == 0x01:
            snap(self)
            sensor_lock = True
        elif value[0] != 0xff and value[2] == 0x00:
            sensor_lock = False
            
#######################################################################
def raw_Q2image_Q(self): # 촬영모드 전용
    while not self.stop_signal:
        time.sleep(0.02)
        
        if self.raw_Q.empty(): continue
        self.image_Q.put(self.raw_Q.get())
                
#######################################################################
def read(self):
    try:
        while not self.stop_signal:
            time.sleep(0.01)

            # get image
            if self.raw_Q.empty(): continue
            img = self.raw_Q.get()
            
            # detect
            start_time = time.time()
            datas, poly_boxes, debug_img = self.code_detector(img)
            end_time = time.time()
            
            logger.info(f"Detect Time : {end_time-start_time:.3f}")
            self.analy_Q.put([img, datas, poly_boxes])

            # save
            self.recode_Q.put([img, 'raw'])
            self.recode_Q.put([debug_img, 'debug'])
            
    except Exception as e:
        logger.error(f"[read]{e}")
        self.write_sys_msg(e)
        self.stop_signal = True
        
#######################################################################
def turn_off(self, data, n_time):
    time.sleep(n_time)
    self.thr_lock.acquire()
    self.serial.write(data)
    self.thr_lock.release()

def analysis(self):
    try:
        while not self.stop_signal:
            time.sleep(0.01)

            # get img, detect
            if self.analy_Q.empty(): continue
            img, datas, poly_boxes = self.analy_Q.get()
            
            # alram
            if not datas:
                self.thr_lock.acquire()
                self.serial.write(BYTES_DIC["red_on"])
                self.thr_lock.release()
                # turn off alram after 0.2s
                Thread(target=turn_off, args=(self, BYTES_DIC["red_off"], 0.2), daemon=True).start()
   
            # select
            code = datas[0] if datas else None
            poly = poly_boxes[0] if poly_boxes else None
            
            # make path for db
            if code: path = os.path.join(SAVE_OK_IMG_DIR, code, f"{tool.get_time_str()}.jpg")
            else: path = os.path.join(SAVE_NG_IMG_DIR, f"{tool.get_time_str()}.jpg")
            
            self.data_Q.put(code)
            self.draw_Q.put([img.copy(), code, poly])
            self.db_Q.put([code, path])
        
    except Exception as e:
        logger.error(f"[analysis]{e}")
        self.write_sys_msg(e)
        self.stop_signal = True
        
#######################################################################
def draw(self):
    fc = lambda x,y:np.random.randint(x,y)
    colors = [(fc(50,255), fc(50,255), fc(0,150)) for _ in range(len(self.code2name))]
    color_dic = dict(zip(self.code2name, colors))
    # font = cv2.FONT_HERSHEY_SIMPLEX
    font = ImageFont.truetype(FONT_PATH, 40)
    
    # img_shape = np.array(self.cam.img_shape[:2])[::-1] # xy
    
    try:
        while not self.stop_signal:
            time.sleep(0.01)

            # get img, names, marker, polys
            if self.draw_Q.empty(): continue
            img, code, poly = self.draw_Q.get()
            name = code
            if poly is not None:
                poly = poly.astype(np.int32)
            

            # draw area box # cv2에서는 BGR이지만 카메라로 촬영한 이미지이기 때문에 (255,0,0) -> Red
            # cv2.rectangle(img, real_area_box[0], real_area_box[1], (255,0,0), 3)

            # draw poly
            color = color_dic[name] if name in color_dic else (255,255,0)
            cv2.polylines(img, [poly], True, color, thickness=5)
            if name is None:
                self.image_Q.put(img)
                self.recode_Q.put([img, name])
                continue

            # draw name
            x,y = poly[0]
            y -= 40
            img_pil = Image.fromarray(img)
            img_draw = ImageDraw.Draw(img_pil)
            img_draw.text((x,y), name, font=font, fill=(*color, 0))
            img = np.array(img_pil)

            self.image_Q.put(img)
            self.recode_Q.put([img, name])
        
    except Exception as e:
        logger.error(f"[draw]{e}")
        self.write_sys_msg(e)
        self.stop_signal = True

