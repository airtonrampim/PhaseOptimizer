from PyQt5 import QtWidgets, QtCore
import about

class AboutWindow(QtWidgets.QDialog):
    def __init__(self, parent):
        super(AboutWindow, self).__init__(parent)

        self.ui = about.Ui_About()
        self.ui.setupUi(self)
        self.setWindowFlags(QtCore.Qt.Dialog | QtCore.Qt.WindowTitleHint)
