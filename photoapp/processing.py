# photoapp/utils/processing.py

import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from .models import Photo
from django.conf import settings
import os
import io
from django.core.files import File
import tempfile
from PIL import Image


def process_image(imgin):

    image = Image.open(imgin)

    gray_image = image.convert('L')

    buffer = io.BytesIO()
    gray_image.save(buffer, format='JPEG')

    buffer.seek(0)


    file_bytes = np.asarray(bytearray(buffer.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

  
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    channel = cv2.extractChannel(lab, 0)
    clahe = cv2.createCLAHE(clipLimit=11.0, tileGridSize=(8, 8))
    lab = cv2.insertChannel(clahe.apply(channel), lab, 0)
    lab = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5), (-1, -1))
    open = cv2.morphologyEx(lab, cv2.MORPH_OPEN, kernel1)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5), (-1, -1))
    close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, kernel2)

    dest = cv2.subtract(lab, close)
    lab1 = cv2.cvtColor(dest, cv2.COLOR_BGR2Lab)
    channel1 = cv2.extractChannel(lab1, 0)
    clahe1 = cv2.createCLAHE(clipLimit=12.0, tileGridSize=(8, 8))
    lab1 = cv2.insertChannel(clahe1.apply(channel1), lab1, 0)
    lab1 = cv2.cvtColor(lab1, cv2.COLOR_Lab2BGR)
    lab1 = cv2.cvtColor(lab1, cv2.COLOR_BGR2GRAY)

    src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, src_mask = cv2.threshold(src_gray, 30, 255, cv2.THRESH_BINARY)
    src_fin = cv2.bitwise_and(lab1, src_mask)
    ret, lab1_t = cv2.threshold(src_fin, 25, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(lab1_t.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros(lab1_t.shape, dtype=np.uint8)

    for c in range(len(contours)):
        area = cv2.contourArea(contours[c])
        if area <= 100:
            drawing = cv2.drawContours(drawing, contours, c, (255, 0, 0), 2, cv2.LINE_8, hierarchy)

    lab1_b = cv2.bitwise_and(src_fin, drawing)
    res, lab1_bt = cv2.threshold(lab1_b, 30, 255, cv2.THRESH_BINARY_INV)
    lab1_bten = cv2.bitwise_not(lab1_bt)

    contours1, hierarchy1 = cv2.findContours(lab1_bten, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    drawing_fin = np.zeros(lab1_bten.shape, dtype=np.uint8)

    for c in range(len(contours1)):
        area_fin = cv2.contourArea(contours1[c])
        peri = cv2.arcLength(contours1[c], True)
        approx = cv2.approxPolyDP(contours1[c], 0.005 * peri, True)
        if approx.size > 3 and area_fin >= 10:
            drawing_fin = cv2.drawContours(drawing_fin, contours1, c, (255, 0, 0), -1, cv2.LINE_8, hierarchy1)

    final = cv2.GaussianBlur(drawing_fin, (9, 9), 0)
    final = cv2.adaptiveThreshold(final, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 6)

    _, buffer = cv2.imencode('.jpg', final)

    buffer = io.BytesIO(buffer.tobytes())

    buffer.seek(0)
    processed_image = File(buffer, name=f"{imgin.name.split('.')[0]}_processed.jpg")

    return processed_image