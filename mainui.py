# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainui.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(653, 439)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton_00 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_00.setGeometry(QtCore.QRect(10, 140, 121, 41))
        self.pushButton_00.setObjectName("pushButton_00")
        self.pushButton_02 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_02.setGeometry(QtCore.QRect(290, 140, 121, 41))
        self.pushButton_02.setObjectName("pushButton_02")
        self.pushButton_03 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_03.setGeometry(QtCore.QRect(430, 140, 121, 41))
        self.pushButton_03.setObjectName("pushButton_03")
        self.pushButton_01 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_01.setGeometry(QtCore.QRect(150, 140, 121, 41))
        self.pushButton_01.setObjectName("pushButton_01")
        self.pushButton_04 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_04.setGeometry(QtCore.QRect(10, 200, 121, 41))
        self.pushButton_04.setObjectName("pushButton_04")
        self.pushButton_05 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_05.setGeometry(QtCore.QRect(150, 200, 121, 41))
        self.pushButton_05.setObjectName("pushButton_05")
        self.pushButton_10 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_10.setGeometry(QtCore.QRect(290, 260, 121, 41))
        self.pushButton_10.setObjectName("pushButton_10")
        self.pushButton_11 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_11.setGeometry(QtCore.QRect(430, 260, 121, 41))
        self.pushButton_11.setObjectName("pushButton_11")
        self.pushButton_start = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_start.setGeometry(QtCore.QRect(430, 30, 121, 41))
        self.pushButton_start.setObjectName("pushButton_start")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(220, 390, 141, 61))
        self.label.setText("")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 30, 271, 41))
        self.label_2.setObjectName("label_2")
        self.pushButton_select = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_select.setGeometry(QtCore.QRect(290, 30, 121, 41))
        self.pushButton_select.setObjectName("pushButton_select")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(10, 100, 181, 31))
        self.label_3.setObjectName("label_3")
        self.pushButton_06 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_06.setGeometry(QtCore.QRect(290, 200, 121, 41))
        self.pushButton_06.setObjectName("pushButton_06")
        self.pushButton_07 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_07.setGeometry(QtCore.QRect(430, 200, 121, 41))
        self.pushButton_07.setObjectName("pushButton_07")
        self.pushButton_08 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_08.setGeometry(QtCore.QRect(10, 260, 121, 41))
        self.pushButton_08.setObjectName("pushButton_08")
        self.pushButton_09 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_09.setGeometry(QtCore.QRect(150, 260, 121, 41))
        self.pushButton_09.setObjectName("pushButton_09")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(10, 340, 631, 31))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(10, 380, 631, 16))
        self.label_5.setObjectName("label_5")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 653, 18))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_00.setText(_translate("MainWindow", "原图像"))
        self.pushButton_02.setText(_translate("MainWindow", "灰度图像"))
        self.pushButton_03.setText(_translate("MainWindow", "高斯滤波"))
        self.pushButton_01.setText(_translate("MainWindow", "调节尺寸"))
        self.pushButton_04.setText(_translate("MainWindow", "sobel求导"))
        self.pushButton_05.setText(_translate("MainWindow", "otsu二值化"))
        self.pushButton_10.setText(_translate("MainWindow", "旋转矫正"))
        self.pushButton_11.setText(_translate("MainWindow", "逐行滤波"))
        self.pushButton_start.setText(_translate("MainWindow", "开始处理"))
        self.label_2.setText(_translate("MainWindow", "请选择要识别的图片"))
        self.pushButton_select.setText(_translate("MainWindow", "选择"))
        self.label_3.setText(_translate("MainWindow", "分步骤展示处理过程："))
        self.pushButton_06.setText(_translate("MainWindow", "取连通域"))
        self.pushButton_07.setText(_translate("MainWindow", "切割图片"))
        self.pushButton_08.setText(_translate("MainWindow", "调整大小"))
        self.pushButton_09.setText(_translate("MainWindow", "开、闭、膨胀"))
        self.label_4.setText(_translate("MainWindow", "使用指南：首先点击‘选择’选择图片，之后点击‘开始处理’得到处理结果。"))
        self.label_5.setText(_translate("MainWindow", "分步骤处理过程中，每个按钮点击一次会显示图片，再次点击会关闭显示"))

