from PyQt5 import QtWidgets, QtCore, QtGui
import sys


class purple(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.draw_ui()
        self.draw_connections()
        self.show()

    def draw_ui(self):
        self.setWindowTitle("Quantum Computer")
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        size_policy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(size_policy)
        self.setMinimumSize(QtCore.QSize(860, 690))
        self.setMaximumSize(QtCore.QSize(860, 690))
        status_window = QtWidgets.QPlainTextEdit(self)
        status_window.setGeometry(10, 480, 840, 200)
        status_window.setReadOnly(True)
        status_window.setLineWrapMode(False)
        status_window.setFont(QtGui.QFont("Courier", 10))
        tab_widget = QtWidgets.QTabWidget(self)
        tab_widget.setGeometry(10, 10, 840, 460)

        self.populate_computer_tab(tab_widget)

    def populate_computer_tab(self, tab_widget):
        sku_tab = QtWidgets.QWidget()
        tab_widget.addTab(sku_tab, "Computer")
        layout = self.create_9by9_tab_layout(sku_tab)
        cells = dict()
        options = ["-----", "H", "X"]
        for i in range(9):
            for j in range(9):
                cells[(i, j)] = QtWidgets.QComboBox()
                cells[(i, j)].addItems(options)
                cells[(i, j)].setFixedHeight(40)
                layout.addWidget(cells[(i, j)], i, j)
                if j == 0:
                    cells[(i, j)].setEnabled(False)


    def draw_connections(self):
        pass

    def say(self, text_string):
        pass

    def lock_interface(self, value):
        pass

    def open_folder(self):
        pass

    @staticmethod
    def create_9by9_tab_layout(parent):
        layout = QtWidgets.QGridLayout(parent)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(10)
        for i in range(9):
            layout.setColumnStretch(i, 1)
            layout.setRowStretch(i, 1)
        return layout


if __name__ == "__main__":
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    main_application = QtWidgets.QApplication(sys.argv)
    main_window = purple()
    sys.exit(main_application.exec_())
