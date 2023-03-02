from utils.logger import logger
from utils.text import *
from utils import tool, device
from PIL import ImageTk, Image
import numpy as np
import time

from tkinter import PhotoImage

def device_check(self):
    # Load Cam
    self.cam, err = device.get_cam(EXPOSURE_TIME, logger=logger)
    if self.cam:
        logger.info("Cam Started.")
        self.write_sys_msg("Cam Started.")

    # Load Serial
    self.serial, err = device.get_serial(SERIAL_PORT)
    if self.serial:
        logger.info("Serial opened.")
        self.write_sys_msg("Serial opened.")
        
    # 에러가 있을 경우 출력
    if err:
        logger.error(err)
        self.write_sys_msg(err)

    text = f"Cam state : {bool(self.cam)}   Serial state : {bool(self.serial)}"
    self.write_sys_msg(text)
    
    if self.cam is not None and self.serial is not None:
        self.init_button_()

# 실시간 이미지 조정####################################################
def image_eater(self): # 쓰레드 # self.image_Q에 있는 이미지 출력
    current_winfo = self.image_frame.winfo_width(), self.image_frame.winfo_height()
    while True:
        time.sleep(0.02)
        if self.stop_signal: break
        last_winfo = self.image_frame.winfo_width(), self.image_frame.winfo_height()
            
        if current_winfo == last_winfo and self.image_Q.empty(): continue
        if current_winfo != last_winfo: current_winfo = last_winfo
        if not self.image_Q.empty(): self.current_origin_image = self.image_Q.get() # BGR
        if self.current_origin_image is None: continue
            
        __auto_resize_img(self)
        imgtk = ImageTk.PhotoImage(Image.fromarray(self.current_image[:,:,::-1]))
        self.image_label.configure(image=imgtk)
        self.image_label.image = imgtk
    
    self.current_origin_image = None #np.zeros((100,100,3), dtype=np.uint8)
    self.current_image = None
    self.image_label.configure(image=None)
    self.image_label.image = None
    

def __auto_resize_img(self):
    h, w = self.current_origin_image.shape[:2]
    ratio = h/w
    wh = self.image_frame.winfo_height() - 24
    ww = self.image_frame.winfo_width() - 24
    wratio = wh/ww
        
    if ratio < wratio: size, target = ww, 'w'
    else: size, target = wh, 'h'
    self.current_image = tool.fix_ratio_resize_img(self.current_origin_image, size=size, target=target)

# 실시간 데이터 수정####################################################
def data_eater(self):
    while True:
        time.sleep(0.02)
        if self.stop_signal: break
        if self.data_Q.empty(): continue

        code = self.data_Q.get()
        self.code2cnt[code] += 1
        update_gui(self, code)

def update_gui(self, code, init=False):
    # day_cnt gui
    day_cnt_all = sum(self.code2cnt.values())
    day_cnt_ng = self.code2cnt[None]
    day_cnt_ok = day_cnt_all - day_cnt_ng
    self.day_cnt_all.configure(text=day_cnt_all)
    self.day_cnt_ok.configure(text=day_cnt_ok)
    self.day_cnt_ng.configure(text=day_cnt_ng)
    if init: return

    # single_cnt
    if code is None: name = "인식실패"
    elif code in self.code2name: name = self.code2name[code]
    else: name = "새로운 제품"
    self.name_label.configure(text=name)
    self.single_cnt.configure(text=self.code2cnt[code])

    # OK, NG
    if code: self.ok_label.configure(text='OK', fg='#6f6')
    else: self.ok_label.configure(text='NG', fg='#f00')
    
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################















