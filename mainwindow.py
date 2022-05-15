# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1241, 524)
        MainWindow.setFocusPolicy(QtCore.Qt.NoFocus)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icon.xpm"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.gbCamera = QtWidgets.QGroupBox(self.centralwidget)
        self.gbCamera.setEnabled(True)
        self.gbCamera.setObjectName("gbCamera")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.gbCamera)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.lblCamera = QtWidgets.QLabel(self.gbCamera)
        self.lblCamera.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
        self.lblCamera.setText("")
        self.lblCamera.setAlignment(QtCore.Qt.AlignCenter)
        self.lblCamera.setObjectName("lblCamera")
        self.gridLayout_4.addWidget(self.lblCamera, 0, 0, 1, 2)
        self.gridLayout.addWidget(self.gbCamera, 1, 1, 1, 1)
        self.gbPhase = QtWidgets.QGroupBox(self.centralwidget)
        self.gbPhase.setEnabled(False)
        self.gbPhase.setObjectName("gbPhase")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.gbPhase)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.pbAlign = QtWidgets.QPushButton(self.gbPhase)
        self.pbAlign.setObjectName("pbAlign")
        self.gridLayout_3.addWidget(self.pbAlign, 8, 0, 1, 3)
        self.lblIntValue = QtWidgets.QLabel(self.gbPhase)
        self.lblIntValue.setEnabled(False)
        self.lblIntValue.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.lblIntValue.setObjectName("lblIntValue")
        self.gridLayout_3.addWidget(self.lblIntValue, 8, 7, 1, 1)
        self.pbCorrect = QtWidgets.QPushButton(self.gbPhase)
        self.pbCorrect.setEnabled(False)
        self.pbCorrect.setObjectName("pbCorrect")
        self.gridLayout_3.addWidget(self.pbCorrect, 8, 9, 1, 1)
        self.line = QtWidgets.QFrame(self.gbPhase)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout_3.addWidget(self.line, 5, 0, 1, 10)
        self.sbLineThickness = QtWidgets.QSpinBox(self.gbPhase)
        self.sbLineThickness.setMaximumSize(QtCore.QSize(52, 32))
        self.sbLineThickness.setMinimum(1)
        self.sbLineThickness.setMaximum(1024)
        self.sbLineThickness.setObjectName("sbLineThickness")
        self.gridLayout_3.addWidget(self.sbLineThickness, 1, 3, 1, 1)
        self.lblLineThickness = QtWidgets.QLabel(self.gbPhase)
        self.lblLineThickness.setObjectName("lblLineThickness")
        self.gridLayout_3.addWidget(self.lblLineThickness, 1, 2, 1, 1)
        self.lblConfidence = QtWidgets.QLabel(self.gbPhase)
        self.lblConfidence.setObjectName("lblConfidence")
        self.gridLayout_3.addWidget(self.lblConfidence, 8, 3, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem, 8, 6, 1, 1)
        self.lblPhase = QtWidgets.QLabel(self.gbPhase)
        self.lblPhase.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
        self.lblPhase.setText("")
        self.lblPhase.setAlignment(QtCore.Qt.AlignCenter)
        self.lblPhase.setObjectName("lblPhase")
        self.gridLayout_3.addWidget(self.lblPhase, 0, 2, 1, 8)
        self.sbIntValue = QtWidgets.QSpinBox(self.gbPhase)
        self.sbIntValue.setEnabled(False)
        self.sbIntValue.setMinimum(1)
        self.sbIntValue.setMaximum(255)
        self.sbIntValue.setProperty("value", 255)
        self.sbIntValue.setObjectName("sbIntValue")
        self.gridLayout_3.addWidget(self.sbIntValue, 8, 8, 1, 1)
        self.gridLayout.addWidget(self.gbPhase, 1, 0, 2, 1)
        self.gbCameraSideView = QtWidgets.QGroupBox(self.centralwidget)
        self.gbCameraSideView.setEnabled(True)
        self.gbCameraSideView.setMaximumSize(QtCore.QSize(16777215, 250))
        self.gbCameraSideView.setObjectName("gbCameraSideView")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gbCameraSideView)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.lblIn = QtWidgets.QLabel(self.gbCameraSideView)
        self.lblIn.setObjectName("lblIn")
        self.gridLayout_2.addWidget(self.lblIn, 0, 2, 1, 1)
        self.cbPosition = QtWidgets.QComboBox(self.gbCameraSideView)
        self.cbPosition.setObjectName("cbPosition")
        self.cbPosition.addItem("")
        self.cbPosition.addItem("")
        self.gridLayout_2.addWidget(self.cbPosition, 0, 1, 1, 1)
        self.lblCameraSideView = QtWidgets.QLabel(self.gbCameraSideView)
        self.lblCameraSideView.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
        self.lblCameraSideView.setText("")
        self.lblCameraSideView.setAlignment(QtCore.Qt.AlignCenter)
        self.lblCameraSideView.setObjectName("lblCameraSideView")
        self.gridLayout_2.addWidget(self.lblCameraSideView, 1, 0, 1, 6)
        self.lblPosition = QtWidgets.QLabel(self.gbCameraSideView)
        self.lblPosition.setObjectName("lblPosition")
        self.gridLayout_2.addWidget(self.lblPosition, 0, 0, 1, 1)
        self.sbPositionValue = QtWidgets.QSpinBox(self.gbCameraSideView)
        self.sbPositionValue.setMinimum(1)
        self.sbPositionValue.setMaximum(1280)
        self.sbPositionValue.setProperty("value", 320)
        self.sbPositionValue.setObjectName("sbPositionValue")
        self.gridLayout_2.addWidget(self.sbPositionValue, 0, 3, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem1, 0, 4, 1, 1)
        self.gridLayout.addWidget(self.gbCameraSideView, 2, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1241, 30))
        self.menubar.setObjectName("menubar")
        self.mnFile = QtWidgets.QMenu(self.menubar)
        self.mnFile.setObjectName("mnFile")
        self.mnHelp = QtWidgets.QMenu(self.menubar)
        self.mnHelp.setObjectName("mnHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actQuit = QtWidgets.QAction(MainWindow)
        self.actQuit.setObjectName("actQuit")
        self.actLoadImage = QtWidgets.QAction(MainWindow)
        self.actLoadImage.setObjectName("actLoadImage")
        self.actAbout = QtWidgets.QAction(MainWindow)
        self.actAbout.setObjectName("actAbout")
        self.actSaveImage = QtWidgets.QAction(MainWindow)
        self.actSaveImage.setObjectName("actSaveImage")
        self.actSavePhase = QtWidgets.QAction(MainWindow)
        self.actSavePhase.setObjectName("actSavePhase")
        self.actSaveGraphic = QtWidgets.QAction(MainWindow)
        self.actSaveGraphic.setObjectName("actSaveGraphic")
        self.actUpdatePhase = QtWidgets.QAction(MainWindow)
        self.actUpdatePhase.setObjectName("actUpdatePhase")
        self.mnFile.addAction(self.actLoadImage)
        self.mnFile.addSeparator()
        self.mnFile.addAction(self.actQuit)
        self.mnHelp.addAction(self.actAbout)
        self.menubar.addAction(self.mnFile.menuAction())
        self.menubar.addAction(self.mnHelp.menuAction())

        self.retranslateUi(MainWindow)
        self.cbPosition.setCurrentIndex(0)
        self.actQuit.triggered.connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.sbLineThickness, self.pbAlign)
        MainWindow.setTabOrder(self.pbAlign, self.pbCorrect)
        MainWindow.setTabOrder(self.pbCorrect, self.cbPosition)
        MainWindow.setTabOrder(self.cbPosition, self.sbPositionValue)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Otimizar padrão de fase"))
        self.gbCamera.setTitle(_translate("MainWindow", "Câmera"))
        self.gbPhase.setTitle(_translate("MainWindow", "Padrão de fase"))
        self.pbAlign.setText(_translate("MainWindow", "A&linhar"))
        self.lblIntValue.setText(_translate("MainWindow", "Intensidade máxima:"))
        self.pbCorrect.setText(_translate("MainWindow", "Corrigir"))
        self.lblLineThickness.setText(_translate("MainWindow", "Espessura da linha:"))
        self.lblConfidence.setText(_translate("MainWindow", "Confidência: 000.00%"))
        self.gbCameraSideView.setTitle(_translate("MainWindow", "Vista lateral"))
        self.lblIn.setText(_translate("MainWindow", "em"))
        self.cbPosition.setItemText(0, _translate("MainWindow", "X"))
        self.cbPosition.setItemText(1, _translate("MainWindow", "Y"))
        self.lblPosition.setText(_translate("MainWindow", "Posição"))
        self.mnFile.setTitle(_translate("MainWindow", "&Arquivo"))
        self.mnHelp.setTitle(_translate("MainWindow", "A&juda"))
        self.actQuit.setText(_translate("MainWindow", "&Sair"))
        self.actLoadImage.setText(_translate("MainWindow", "&Carregar imagem"))
        self.actAbout.setText(_translate("MainWindow", "Sobre"))
        self.actSaveImage.setText(_translate("MainWindow", "Salvar imagem"))
        self.actSaveImage.setToolTip(_translate("MainWindow", "Salvar imagem"))
        self.actSavePhase.setText(_translate("MainWindow", "Salvar fase"))
        self.actSaveGraphic.setText(_translate("MainWindow", "Salvar gráfico"))
        self.actUpdatePhase.setText(_translate("MainWindow", "Atualizar"))
