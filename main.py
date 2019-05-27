# -*- coding: UTF-8 -*-
import math
import sys
import tkinter as tk
from tkinter import filedialog
import cv2 as cv
import numpy as np
from PIL import Image
from PyQt5 import QtWidgets
from scipy import ndimage
from mainui import Ui_MainWindow
import os
import pandas as pd


show_dic = {}  # 这个里面放着所有的要展示的阶段性处理图像
file_selected = ''


class var:
    close0 = 0
    close1 = 0
    close2 = 0
    close3 = 0
    close4 = 0
    close5 = 0
    close6 = 0
    close7 = 0
    close8 = 0
    close9 = 0
    close10 = 0
    close11 = 0


class mywindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(mywindow, self).__init__()
        self.setupUi(self)
        self.initui()
        self.pushButton_select.clicked.connect(self.choosePhoto)
        self.pushButton_start.clicked.connect(self.numberreadercode)

    def choosePhoto(self):
        root = tk.Tk()
        root.withdraw()
        global file_selected
        file_selected = filedialog.askopenfilename()
        # file_selected= file_selected.decode("utf-8").encode("gbk")
        print('aa')

    def cv_imread(self,file_path):
        cv_img = cv.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)

        return cv_img



    def numberreadercode(self):

        img = self.cv_imread(file_selected)

        show_dic[0] = img  # 0 原始图像
        (x, y, _) = img.shape
        x_s = 400  # 计算调整的大小至宽为400
        y_s = int(y * x_s / x)  # 宽度按比例变化
        out_resize = cv.resize(img, (y_s, x_s))  # 调整大小后的输出
        show_dic[1] = out_resize  # 1调整大小的图像
        show_dic[1] = out_resize

        gray = cv.cvtColor(out_resize, cv.COLOR_BGR2GRAY)  # 灰化处理图像
        show_dic[2] = gray  # 2灰化图像

        smooth = cv.GaussianBlur(gray, (3, 3), 0)  # 高斯滤波平滑处理图像
        show_dic[3] = smooth  # 3 滤波后的灰度图

        sobel1 = cv.Sobel(smooth, -1, 2, 0)  # x方向求导
        sobel2 = cv.Sobel(smooth, -1, 0, 2)  # y方向求导
        sobel = cv.addWeighted(sobel1, 1, sobel2, 1, 0)  # 两方向叠加
        show_dic[4] = sobel  # 4 sobel算子操作 取边界

        ret1, th1 = cv.threshold(sobel, 0, 255, cv.THRESH_OTSU)  # otsu二值化
        show_dic[5] = th1  # 5 ostu

        a, contours, hierarchy = cv.findContours(th1, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)  # 取连通域
        contours_max = 0
        max_i = 0
        for i in range(len(contours)):  # 取矩形面积最大的连通域
            x, y, w, h = cv.boundingRect(contours[i])
            if (w / h) > 1.4 and (w / h) < 2.5:
                if w * h > contours_max:
                    contours_max = w * h
                    max_i = i

        x_m, y_m, w_m, h_m = cv.boundingRect(contours[max_i])
        tocut = out_resize.copy()  # 复制一个原图像以备切割
        cv.rectangle(tocut, (x_m, y_m), (x_m + w_m, y_m + h_m), (150, 150, 0), 2)  # 在图像中画出选中连通域的矩形框
        show_dic[6] = tocut  # 6标出矩形框的图像

        # 切割轮廓
        aftercut = tocut[y_m + 15:y_m + h_m - 15, x_m + 10:x_m + w_m - 10]  # 先用y确定高，再用x确定宽
        show_dic[7] = aftercut  # 7刚剪切完的图像

        x_s = 200
        y_s = 100  # int(y*x_s/x)
        cut_resize = cv.resize(aftercut, (x_s, y_s), Image.ANTIALIAS)
        show_dic[8] = cut_resize  # 8调整大小后的图像

        # 图像形态学操作

        gray_cut = cv.cvtColor(cut_resize, cv.COLOR_BGR2GRAY)  # 灰化处理图像

        smooth_cut = cv.GaussianBlur(gray_cut, (3, 3), 0)  # 平滑处理图像

        sobel1 = cv.Sobel(smooth_cut, -1, 2, 0)  # x方向求导
        sobel2 = cv.Sobel(smooth_cut, -1, 0, 2)  # y方向求导
        done_cut = cv.addWeighted(sobel1, 1, sobel2, 1, 0)  # 两方向叠加

        smooth_cut = cv.GaussianBlur(done_cut, (3, 3), 0)  # 平滑处理图像

        ret1, th1_cut = cv.threshold(smooth_cut, 0, 255, cv.THRESH_OTSU)  # otsu二值化

        kernel_2 = np.ones((2, 2), np.uint8)  # 核
        kernel_3 = np.ones((3, 3), np.uint8)  # 核

        opening2_cut = cv.morphologyEx(th1_cut, cv.MORPH_OPEN, kernel_2, iterations=1)  # 图像开运算
        closing_cut = cv.morphologyEx(opening2_cut, cv.MORPH_CLOSE, kernel_3, iterations=1)  # 图像闭运算
        dilation = cv.dilate(closing_cut, kernel_2, iterations=1)  # 图像膨胀
        # 操作完了
        show_dic[9] = dilation  # 9开闭

        # 对处理完的图像进行旋转调整
        lines = cv.HoughLines(dilation, 1, np.pi / 180, 0)
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            if x1 == x2 or y1 == y2:
                continue
            t = float(y2 - y1) / (x2 - x1)
            rotate_angle = math.degrees(math.atan(t))
            if rotate_angle > 45:
                rotate_angle = -90 + rotate_angle
            elif rotate_angle < -45:
                rotate_angle = 90 + rotate_angle
        rotate_img = ndimage.rotate(dilation, rotate_angle)
        show_dic[10] = rotate_img  # 10旋转

        # 对旋转完的图像进行滤波 去除小块块
        [rows, cols] = rotate_img.shape
        unnoise = rotate_img.copy()
        for i_clean_y in range(rows):
            if unnoise.sum(1)[i_clean_y] < 8000:
                for j_clean in range(cols):
                    unnoise[i_clean_y][j_clean] = 0

        for i_clean_x in range(cols):
            if unnoise.sum(0)[i_clean_x] < 1100:
                for j_clean in range(rows):
                    unnoise[j_clean][i_clean_x] = 0

        show_dic[11] = unnoise  # 11滤掉小杂点的图像

        cv.imshow('done', unnoise)
        cv.imwrite('output.jpg', unnoise)
        cv.waitKey(0)

    def initui(self):
        self.pushButton_00.clicked.connect(self.pushbutton0)
        self.pushButton_01.clicked.connect(self.pushbutton1)
        self.pushButton_02.clicked.connect(self.pushbutton2)
        self.pushButton_03.clicked.connect(self.pushbutton3)
        self.pushButton_04.clicked.connect(self.pushbutton4)
        self.pushButton_05.clicked.connect(self.pushbutton5)
        self.pushButton_06.clicked.connect(self.pushbutton6)
        self.pushButton_07.clicked.connect(self.pushbutton7)
        self.pushButton_08.clicked.connect(self.pushbutton8)
        self.pushButton_09.clicked.connect(self.pushbutton9)
        self.pushButton_10.clicked.connect(self.pushbutton10)
        self.pushButton_11.clicked.connect(self.pushbutton11)

    @staticmethod
    def pushbutton0(self):
        if (var.close0 == 0):
            # cv.namedWindow('   ', cv.WINDOW_AUTOSIZE)
            cv.imshow('origin', show_dic[0])
            var.close0 = 1
        else:
            cv.destroyWindow('origin')
            var.close0 = 0

    @staticmethod
    def pushbutton1(self):
        if (var.close1 == 0):
            # cv.namedWindow('   ', cv.WINDOW_AUTOSIZE)
            cv.imshow('resize', show_dic[1])
            var.close1 = 1
        else:
            cv.destroyWindow('resize')
            var.close1 = 0

    @staticmethod
    def pushbutton2(self):
        if (var.close2 == 0):
            cv.imshow('gray', show_dic[2])
            var.close2 = 1
        else:
            cv.destroyWindow('gray')
            var.close2 = 0

    @staticmethod
    def pushbutton3(self):
        if (var.close3 == 0):
            cv.imshow('Gaussion', show_dic[3])
            var.close3 = 1
        else:
            cv.destroyWindow('Gaussion')
            var.close3 = 0

    @staticmethod
    def pushbutton4(self):
        if (var.close4 == 0):
            # cv.namedWindow(' ', cv.WINDOW_AUTOSIZE)
            cv.imshow('sobel', show_dic[4])
            var.close4 = 1
        else:
            cv.destroyWindow('sobel')
            var.close4 = 0

    @staticmethod
    def pushbutton5(self):
        if (var.close5 == 0):
            # cv.namedWindow('  ', cv.WINDOW_AUTOSIZE)
            cv.imshow('ostu', show_dic[5])
            var.close5 = 1
        else:
            cv.destroyWindow('ostu')
            var.close5 = 0

    @staticmethod
    def pushbutton6(self):
        if (var.close6 == 0):
            # cv.namedWindow('  ', cv.WINDOW_AUTOSIZE)
            cv.imshow('connection', show_dic[6])
            var.close6 = 1
        else:
            cv.destroyWindow('connection')
            var.close6 = 0

    @staticmethod
    def pushbutton7(self):
        if (var.close7 == 0):
            # cv.namedWindow('  ', cv.WINDOW_AUTOSIZE)
            cv.imshow('cut', show_dic[7])
            var.close7 = 1
        else:
            cv.destroyWindow('cut')
            var.close7 = 0

    @staticmethod
    def pushbutton8(self):
        if (var.close8 == 0):
            # cv.namedWindow('  ', cv.WINDOW_AUTOSIZE)
            cv.imshow('resize2', show_dic[8])
            var.close8 = 1
        else:
            cv.destroyWindow('resize2')
            var.close8 = 0

    @staticmethod
    def pushbutton9(self):
        if (var.close9 == 0):
            # cv.namedWindow('  ', cv.WINDOW_AUTOSIZE)
            cv.imshow('openclose', show_dic[9])
            var.close9 = 1
        else:
            cv.destroyWindow('openclose')
            var.close9 = 0

    @staticmethod
    def pushbutton10(self):
        if (var.close10 == 0):
            # cv.namedWindow('  ', cv.WINDOW_AUTOSIZE)
            cv.imshow('rotate', show_dic[10])
            var.close10 = 1
        else:
            cv.destroyWindow('rotate')
            var.close10 = 0

    @staticmethod
    def pushbutton11(self):
        if (var.close11 == 0):
            # cv.namedWindow('  ', cv.WINDOW_AUTOSIZE)
            cv.imshow('unnoise', show_dic[11])
            var.close11 = 1
        else:
            cv.destroyWindow('unnoise')
            var.close11 = 0


# erosion = cv.erode(src, kernel)      #图像腐蚀
# dilation = cv.dilate(src, kernel)   #图像膨胀
# closing = cv.morphologyEx(src, cv.MORPH_CLOSE, kernel, ) #图像闭运算
# opening = cv.morphologyEx(src, cv.MORPH_OPEN, kernel, iterations=1)#图像开运算

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = mywindow()
    MainWindow.show()
    sys.exit(app.exec_())
