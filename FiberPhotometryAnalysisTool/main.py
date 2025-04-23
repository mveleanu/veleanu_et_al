from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
import sys
import numpy as np
from screens.analysis_screen import AnalysisScreen

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.window = None
        self.button = QPushButton('Start Analysis')
        self.setCentralWidget(self.button)

        self.button.clicked.connect(self.start_analysis)

    
    def start_analysis(self, checked):
        if self.window is None:
            self.window = AnalysisScreen()
            self.window.show()
        else:
            self.window.close()
            self.window = None



app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())