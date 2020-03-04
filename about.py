# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'about.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_About(object):
    def setupUi(self, About):
        About.setObjectName("About")
        About.setWindowModality(QtCore.Qt.WindowModal)
        About.resize(400, 300)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(About.sizePolicy().hasHeightForWidth())
        About.setSizePolicy(sizePolicy)
        About.setMinimumSize(QtCore.QSize(400, 300))
        About.setMaximumSize(QtCore.QSize(400, 300))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icon.xpm"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        About.setWindowIcon(icon)
        About.setModal(True)
        self.lblFigure = QtWidgets.QLabel(About)
        self.lblFigure.setGeometry(QtCore.QRect(140, 10, 121, 101))
        self.lblFigure.setText("")
        self.lblFigure.setPixmap(QtGui.QPixmap("icon.xpm"))
        self.lblFigure.setScaledContents(True)
        self.lblFigure.setObjectName("lblFigure")
        self.teDescription = QtWidgets.QTextEdit(About)
        self.teDescription.setGeometry(QtCore.QRect(10, 120, 381, 171))
        self.teDescription.setReadOnly(True)
        self.teDescription.setObjectName("teDescription")

        self.retranslateUi(About)
        QtCore.QMetaObject.connectSlotsByName(About)

    def retranslateUi(self, About):
        _translate = QtCore.QCoreApplication.translate
        About.setWindowTitle(_translate("About", "Sobre"))
        self.teDescription.setHtml(_translate("About", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Noto Sans\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">Sobre o programa</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p align=\"justify\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Lorem ipsum dolor sit amet, consectetur adipiscing elit. Proin imperdiet odio in eros volutpat, quis consectetur ligula molestie. In nec tempor tortor, gravida pellentesque purus. In at porttitor sem, non iaculis ligula.</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p align=\"justify\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Nunc congue leo vitae ante tincidunt, sed fermentum ex lacinia. Nullam euismod quam id ipsum tincidunt, nec eleifend ante iaculis. Nullam varius sed tortor sit amet tempor. Sed condimentum enim ac ligula venenatis sodales. Duis ac purus laoreet, dictum neque at, aliquet tellus. </p></body></html>"))
