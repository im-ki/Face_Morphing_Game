from PIL import Image
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QGridLayout, QSpacerItem
from PyQt5.QtGui import QPixmap, QFont, QImage, QPalette, QPainter
import sys
from os import path as os_path
from cv2 import VideoCapture, resize, CAP_DSHOW, INTER_CUBIC, cvtColor, COLOR_BGR2RGB
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QTimer, QEvent, QObject
import numpy as np
from time import sleep
from root_path import pyinstaller_root

from face_obj import Face

img_size = 672

class Camera:
    def __init__(self):
        self.cam = VideoCapture(0+CAP_DSHOW)
        ret, cv_img = self.cam.read()
        self.cv_img = cv_img[:, ::-1, :]
        self.num_h, self.num_w, self.num_c = self.cv_img.shape
        assert ret
        self.cam.release()
        self.state = False

    def open(self):
        if not self.state:
            self.cam = VideoCapture(0+CAP_DSHOW)
            self.state = True

    def read(self):
        assert self.state
        ret, cv_img = self.cam.read()
        cv_img = cv_img[:, ::-1, :]
        assert ret
        self.cv_img = cv_img
        return cv_img

    def close(self):
        if self.state:
            self.cam.release()
            self.state = False
        


class QSSLoader:
    def __init__(self):
        pass

    @staticmethod
    def read_qss_file(qss_file_name):
        with open(qss_file_name, 'r',  encoding='UTF-8') as file:
            return file.read()

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    captured_left_image = pyqtSignal(np.ndarray)
    captured_right_image = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.pause = True
        self.processed = False
        self.left = None

        self.initial = True
        self.draw_initial = True
        init_img = os_path.join(pyinstaller_root, "./data/initial.png")
        self.init_img = np.array(Image.open(init_img))

    def run(self):
        # capture from web cam
        cam = Camera()
        #cap = cv2.VideoCapture("./out.mp4")

        while self._run_flag:
            if self.initial:
                if cam.state:
                    cam.close()
                if self.draw_initial:
                    h, w = cam.num_h, cam.num_w
                    if self.init_img.shape[0] != h or self.init_img.shape[1] != w:
                        self.init_img = resize(self.init_img, (w, h), interpolation=INTER_CUBIC)
                    self.change_pixmap_signal.emit(self.init_img)
                    self.draw_initial = False
                    self.processed = True
                sleep(0.1)
                continue

            if self.pause:
                if not self.processed:
                    if self.left:
                        self.captured_left_image.emit(cam.cv_img)
                    else:
                        self.captured_right_image.emit(cam.cv_img)
                    self.processed = True
                    cam.close()
                sleep(0.1)
                continue
            sleep(0.02)
            if self.processed:
                cam.open()
            self.processed = False
            cv_img = cam.read()
            self.change_pixmap_signal.emit(cv_img)
                    
        # shut down capture system
        cam.close()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

    def init(self):
        self.initial = True
        self.draw_initial = True

        self.pause = True
        self.processed = True

    def Pause(self, left = 0):
        if self.pause == False:
            self.pause = True
            self.left = left
            self.initial = False
            return True
        else:
            return False

    def Continue(self):
        if self.pause:
            self.pause = False
            self.initial = False
            return True
        else:
            return False


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Morphing")
        self.disply_width = 1280
        self.display_height = 960

        outer_layout = QHBoxLayout()
        vbox_left = QVBoxLayout()
        l_grid1 = QGridLayout()
        l_grid2 = QGridLayout()
        vbox_right = QVBoxLayout()
        r_grid1 = QGridLayout()
        r_grid2 = QGridLayout()
        vbox_mid = QVBoxLayout()
        vbox_cam = QVBoxLayout()
        vbox_slider_bar = QVBoxLayout()#QGridLayout()

        h_Spacer_01 = QSpacerItem(20, 2)
        h_Spacer_12 = QSpacerItem(20, 2)
        outer_layout.addLayout(vbox_left)
        outer_layout.addItem(h_Spacer_01)
        outer_layout.addLayout(vbox_mid)
        outer_layout.addItem(h_Spacer_12)
        outer_layout.addLayout(vbox_right)
        vbox_left.addLayout(l_grid1)
        vbox_left.addLayout(l_grid2)
        vbox_right.addLayout(r_grid1)
        vbox_right.addLayout(r_grid2)
        #outer_layout.setStretchFactor(vbox_right, 3)
        # set the vbox layout as the widgets layout


        # create the label that holds the image
        self.image_label = QLabel(self)
        #self.image_label.resize(int(self.disply_width / 3), int(self.display_height / 3))
        ## create a text label
        #self.textLabel = QLabel('Webcam')

        # create a vertical box layout and add the two labels
        #vbox_cam.setAlignment(Qt.AlignHCenter)
        self.reset_button = QPushButton("重置")
        self.reset_button.setFont(QFont('KaiTi', 20))
        self.reset_button.clicked.connect(self.reset)
        self.swap_button = QPushButton("交換")
        self.swap_button.setFont(QFont('KaiTi', 20))
        self.swap_button.clicked.connect(self.swap)
        v_Spacer1_m = QSpacerItem(self.reset_button.sizeHint().width(), 3*self.reset_button.sizeHint().height())
        v_Spacer2_m = QSpacerItem(self.reset_button.sizeHint().width(), 2*self.reset_button.sizeHint().height())
        vbox_cam.addItem(v_Spacer1_m)
        vbox_cam.addWidget(self.image_label)
        vbox_cam.addItem(v_Spacer2_m)
        vbox_cam.addWidget(self.swap_button)
        vbox_cam.addWidget(self.reset_button)
        vbox_cam.setAlignment(Qt.AlignCenter)

        vbox_mid.addLayout(vbox_cam)
        vbox_mid.addLayout(vbox_slider_bar)
#        m_grid = QGridLayout()
        self.slider1 = QSlider(Qt.Horizontal)
        self.slider1.setMinimum(0)
        self.slider1.setMaximum(30)
        self.slider1.setSingleStep(1)
        self.slider1.sliderMoved.connect(self.slider1_func)
        self.text1 = QLabel("改變形狀")
        self.text1.setFont(QFont('KaiTi', 15))
        self.slider2 = QSlider(Qt.Horizontal)
        self.slider2.setMinimum(0)
        self.slider2.setMaximum(30)
        self.slider2.setSingleStep(1)
        self.slider2.sliderMoved.connect(self.slider2_func)
        self.text2 = QLabel("改變表情")
        self.text2.setFont(QFont('KaiTi', 15))
        self.slider3 = QSlider(Qt.Horizontal)
        self.slider3.setMinimum(0)
        self.slider3.setMaximum(30)
        self.slider3.setSingleStep(1)
        self.slider3.sliderMoved.connect(self.slider3_func)
        self.text3 = QLabel("改變顔色")
        self.text3.setFont(QFont('KaiTi', 15))

        vbox_slider_bar.addWidget(self.slider1)
        vbox_slider_bar.addWidget(self.text1)
        vbox_slider_bar.addWidget(self.slider2)
        vbox_slider_bar.addWidget(self.text2)
        vbox_slider_bar.addWidget(self.slider3)
        vbox_slider_bar.addWidget(self.text3)
        vbox_slider_bar.setAlignment(Qt.AlignVCenter)
        vbox_mid.setStretchFactor(vbox_cam, 3)
        vbox_mid.setStretchFactor(vbox_slider_bar, 1)

        self.image1 = QLabel(self)
        self.h_slider_image1 = QSlider(Qt.Horizontal)
        self.h_slider_image1.setMinimum(0)
        self.h_slider_image1.setMaximum(500)
        self.h_slider_image1.setSingleStep(1)
        self.h_slider_image1.setValue(int(self.h_slider_image1.maximum()/2))
        self.h_slider_image1.sliderMoved.connect(self.image1_h_rot)
        h_slider_style_file = os_path.join(pyinstaller_root, './stylesheet/h_rot_slider.qss')
        h_slider_style_sheet = QSSLoader.read_qss_file(h_slider_style_file)
        self.h_slider_image1.setStyleSheet(h_slider_style_sheet)
        self.v_slider_image1 = QSlider(Qt.Vertical)
        self.v_slider_image1.setMinimum(0)
        self.v_slider_image1.setMaximum(500)
        self.v_slider_image1.setSingleStep(1)
        self.v_slider_image1.setValue(int(self.v_slider_image1.maximum()/2))
        self.v_slider_image1.sliderMoved.connect(self.image1_v_rot)
        v_slider_style_file = os_path.join(pyinstaller_root, './stylesheet/v_rot_slider.qss')
        v_slider_style_sheet = QSSLoader.read_qss_file(v_slider_style_file)
        self.v_slider_image1.setStyleSheet(v_slider_style_sheet)
        l_grid1.addWidget(self.image1, 0, 1, 5, 5)
        l_grid1.addWidget(self.h_slider_image1, 5, 1, 1, 5)
        l_grid1.addWidget(self.v_slider_image1, 0, 6, 5, 1)
        horizontalSpacer_l = QSpacerItem(self.v_slider_image1.sizeHint().width(), self.v_slider_image1.sizeHint().height())
        l_grid1.addItem(horizontalSpacer_l, 0, 0, 6, 1)
        l_grid1.setAlignment(Qt.AlignCenter)
        self.b_cap_l = QPushButton("拍攝")#"Capture")
        self.b_cap_l.setFont(QFont('KaiTi', 20))
        self.b_cap_l_status = "pause"#"繼續"#"Continue"
        self.b_cap_l.clicked.connect(self.capture_image_l)
        self.tex_button_l = QPushButton("人面")
        self.tex_button_l.setFont(QFont('KaiTi', 20))
        self.tex_button_l.clicked.connect(lambda: self.change_tex(self.obj1.human_tex, 1))
        self.tex1_button_l = QPushButton("貓面")
        self.tex1_button_l.setFont(QFont('KaiTi', 20))
        self.tex1_button_l.clicked.connect(lambda: self.change_tex(self.cat, 2))
        self.tex2_button_l = QPushButton("虎面")
        self.tex2_button_l.setFont(QFont('KaiTi', 20))
        self.tex2_button_l.clicked.connect(lambda: self.change_tex(self.tiger, 3))
        self.exp_button_l = QPushButton("原表情")
        self.exp_button_l.setFont(QFont('KaiTi', 20))
        self.exp_button_l.clicked.connect(lambda: self.change_exp(self.obj1.org_exp, 1))
        self.exp1_button_l = QPushButton("表情1")
        self.exp1_button_l.setFont(QFont('KaiTi', 20))
        self.exp1_button_l.clicked.connect(lambda: self.change_exp(self.exp1, 2))
        self.exp2_button_l = QPushButton("表情2")
        self.exp2_button_l.setFont(QFont('KaiTi', 20))
        self.exp2_button_l.clicked.connect(lambda: self.change_exp(self.exp2, 3))
        v_Spacer_l = QSpacerItem(self.b_cap_l.sizeHint().width(), self.b_cap_l.sizeHint().height())
        l_grid2.addWidget(self.b_cap_l, 0, 0, 1, 6)
        l_grid2.addItem(v_Spacer_l, 1, 0, 1, 6)
        l_grid2.addWidget(self.tex_button_l, 2, 0, 1, 2)
        l_grid2.addWidget(self.tex1_button_l, 2, 2, 1, 2)
        l_grid2.addWidget(self.tex2_button_l, 2, 4, 1, 2)
        l_grid2.addWidget(self.exp_button_l, 3, 0, 1, 2)
        l_grid2.addWidget(self.exp1_button_l, 3, 2, 1, 2)
        l_grid2.addWidget(self.exp2_button_l, 3, 4, 1, 2)
        l_grid2.setAlignment(Qt.AlignVCenter)
        vbox_left.setStretchFactor(l_grid1, 3)
        vbox_left.setStretchFactor(l_grid2, 1)


        #grid2 = QGridLayout()
        #grid2.setRowStretch(0, 1)
        #grid2.setAlignment(Qt.AlignCenter)
        #grid2.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        self.image2 = QLabel(self)
        self.h_slider_image2 = QSlider(Qt.Horizontal)
        self.h_slider_image2.setMinimum(0)
        self.h_slider_image2.setMaximum(50)
        self.h_slider_image2.setSingleStep(1)
        self.h_slider_image2.setValue(int(self.h_slider_image2.maximum()/2))
        self.h_slider_image2.sliderMoved.connect(self.image2_h_rot)
        self.h_slider_image2.setStyleSheet(h_slider_style_sheet)
        self.v_slider_image2 = QSlider(Qt.Vertical)
        self.v_slider_image2.setMinimum(0)
        self.v_slider_image2.setMaximum(50)
        self.v_slider_image2.setSingleStep(1)
        self.v_slider_image2.setValue(int(self.v_slider_image2.maximum()/2))
        self.v_slider_image2.sliderMoved.connect(self.image2_v_rot)
        self.v_slider_image2.setStyleSheet(v_slider_style_sheet)
        r_grid1.addWidget(self.image2, 0, 1, 5, 5)
        r_grid1.addWidget(self.h_slider_image2, 5, 1, 1, 5)
        r_grid1.addWidget(self.v_slider_image2, 0, 6, 5, 1)
        horizontalSpacer_r = QSpacerItem(self.v_slider_image2.sizeHint().width(), self.v_slider_image2.sizeHint().height())
        r_grid1.addItem(horizontalSpacer_r, 0, 0, 6, 1)
        r_grid1.setAlignment(Qt.AlignCenter)
        self.b_cap_r = QPushButton("拍攝")#"Capture")
        self.b_cap_r.setFont(QFont('KaiTi', 20))
        self.b_cap_r_status = "pause" #"繼續"#"Continue"
        self.b_cap_r.clicked.connect(self.capture_image_r)
        self.tex_button_r = QPushButton("人面")
        self.tex_button_r.setFont(QFont('KaiTi', 20))
        self.tex_button_r.clicked.connect(lambda: self.change_tex(self.obj2.human_tex, 1, left = False))
        self.tex1_button_r = QPushButton("貓面")
        self.tex1_button_r.setFont(QFont('KaiTi', 20))
        self.tex1_button_r.clicked.connect(lambda: self.change_tex(self.cat, 2, left = False))
        self.tex2_button_r = QPushButton("虎面")
        self.tex2_button_r.setFont(QFont('KaiTi', 20))
        self.tex2_button_r.clicked.connect(lambda: self.change_tex(self.tiger, 3, left = False))
        self.exp_button_r = QPushButton("原表情")
        self.exp_button_r.setFont(QFont('KaiTi', 20))
        self.exp_button_r.clicked.connect(lambda: self.change_exp(self.obj2.org_exp, 1, left = False))
        self.exp1_button_r = QPushButton("表情1")
        self.exp1_button_r.setFont(QFont('KaiTi', 20))
        self.exp1_button_r.clicked.connect(lambda: self.change_exp(self.exp1, 2, left = False))
        self.exp2_button_r = QPushButton("表情2")
        self.exp2_button_r.setFont(QFont('KaiTi', 20))
        self.exp2_button_r.clicked.connect(lambda: self.change_exp(self.exp2, 3, left = False))
        v_Spacer_r = QSpacerItem(self.b_cap_r.sizeHint().width(), self.b_cap_r.sizeHint().height())
        r_grid2.addWidget(self.b_cap_r, 0, 0, 1, 6)
        r_grid2.addItem(v_Spacer_r, 1, 0, 1, 6)
        r_grid2.addWidget(self.tex_button_r, 2, 0, 1, 2)
        r_grid2.addWidget(self.tex1_button_r, 2, 2, 1, 2)
        r_grid2.addWidget(self.tex2_button_r, 2, 4, 1, 2)
        r_grid2.addWidget(self.exp_button_r, 3, 0, 1, 2)
        r_grid2.addWidget(self.exp1_button_r, 3, 2, 1, 2)
        r_grid2.addWidget(self.exp2_button_r, 3, 4, 1, 2)
        r_grid2.setAlignment(Qt.AlignVCenter)
        vbox_right.setStretchFactor(r_grid1, 3)
        vbox_right.setStretchFactor(r_grid2, 1)




        #vbox_left.addWidget(self.b_cap)
        #vbox_left.setAlignment(Qt.AlignVCenter)

        #vbox_two_image = QHBoxLayout()


        #grid1.setAlignment(Qt.AlignVCenter)
        #grid2.setAlignment(Qt.AlignVCenter)

        #vbox_two_image.addLayout(grid1)#addWidget(self.image1)
        #vbox_two_image.addLayout(grid2)#addWidget(self.image2)

        ##vbox_two_image.setAlignment(Qt.AlignVCenter)
        #vbox_two_image.setAlignment(Qt.AlignHCenter)

        
        self.setLayout(outer_layout)

        #self.setStyleSheet("MainWindow {background : url(bluetri.png)}")#;)


        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.captured_left_image.connect(self.change_left_image)
        self.thread.captured_right_image.connect(self.change_right_image)
        # start the thread
        self.thread.start()

        # Install event filter for widgets
        self.slider1.installEventFilter(self)
        self.slider2.installEventFilter(self)
        self.slider3.installEventFilter(self)
        self.h_slider_image1.installEventFilter(self)
        self.v_slider_image1.installEventFilter(self)
        self.h_slider_image2.installEventFilter(self)
        self.v_slider_image2.installEventFilter(self)

        self.reset_button.installEventFilter(self)
        self.swap_button.installEventFilter(self)
        
        self.b_cap_l.installEventFilter(self)
        self.tex_button_l.installEventFilter(self)
        self.tex1_button_l.installEventFilter(self)
        self.tex2_button_l.installEventFilter(self)
        self.exp_button_l.installEventFilter(self)
        self.exp1_button_l.installEventFilter(self)
        self.exp2_button_l.installEventFilter(self)
        
        self.b_cap_r.installEventFilter(self)
        self.tex_button_r.installEventFilter(self)
        self.tex1_button_r.installEventFilter(self)
        self.tex2_button_r.installEventFilter(self)
        self.exp_button_r.installEventFilter(self)
        self.exp1_button_r.installEventFilter(self)
        self.exp2_button_r.installEventFilter(self)
        
        self.image1.installEventFilter(self)
        self.image_label.installEventFilter(self)
        self.image2.installEventFilter(self)

        # Set Mouse Checking for Widgets
        self.slider1.setMouseTracking(True)
        self.slider2.setMouseTracking(True)
        self.slider3.setMouseTracking(True)
        self.h_slider_image1.setMouseTracking(True)
        self.v_slider_image1.setMouseTracking(True)
        self.h_slider_image2.setMouseTracking(True)
        self.v_slider_image2.setMouseTracking(True)

        self.reset_button.setMouseTracking(True)
        self.swap_button.setMouseTracking(True)

        self.b_cap_l.setMouseTracking(True)
        self.tex_button_l.setMouseTracking(True)
        self.tex1_button_l.setMouseTracking(True)
        self.tex2_button_l.setMouseTracking(True)
        self.exp_button_l.setMouseTracking(True)
        self.exp1_button_l.setMouseTracking(True)
        self.exp2_button_l.setMouseTracking(True)

        self.b_cap_r.setMouseTracking(True)
        self.tex_button_r.setMouseTracking(True)
        self.tex1_button_r.setMouseTracking(True)
        self.tex2_button_r.setMouseTracking(True)
        self.exp_button_r.setMouseTracking(True)
        self.exp1_button_r.setMouseTracking(True)
        self.exp2_button_r.setMouseTracking(True)

        self.image1.setMouseTracking(True)
        self.image_label.setMouseTracking(True)
        self.image2.setMouseTracking(True)


        #self.setStyleSheet("QWidget {background-image: url(./wave.svg) }")
        #self.setStyleSheet("background-image: ./wave.svg")

        #self.background = QPixmap("./wave.svg")
        self.background = QPixmap(os_path.join(pyinstaller_root, "./data/bluetri.png"))

        slider_style_file = os_path.join(pyinstaller_root, './stylesheet/morph_slider.qss')
        morph_slider_style_sheet = QSSLoader.read_qss_file(slider_style_file)

        morph_slider1_style_sheet = morph_slider_style_sheet.replace("slider_image_path", os_path.join(pyinstaller_root, 'stylesheet/smile.png')).replace('\\', '/')
        self.slider1.setStyleSheet(morph_slider1_style_sheet)

        morph_slider2_style_sheet = morph_slider_style_sheet.replace("slider_image_path", os_path.join(pyinstaller_root, 'stylesheet/grin.png')).replace('\\', '/')
        self.slider2.setStyleSheet(morph_slider2_style_sheet)

        morph_slider3_style_sheet = morph_slider_style_sheet.replace("slider_image_path", os_path.join(pyinstaller_root, 'stylesheet/ch_color.png')).replace('\\', '/')
        self.slider3.setStyleSheet(morph_slider3_style_sheet)

        self.tex_state_l = 1
        self.exp_state_l = 1
        self.tex_state_r = 1
        self.exp_state_r = 1

        # For auto playing
        self.s1_direction = 1
        self.s2_direction = 1
        self.s3_direction = 1
    
        ##########################################################
        # Application
        ##########################################################
        mask_folder = os_path.join(pyinstaller_root, "./facial_mask")
        self.cat = np.load(os_path.join(mask_folder, "cat.npy")) / 255
        self.tiger = np.load(os_path.join(mask_folder, "tiger.npy")) / 255
        #self.cat = torch.from_numpy(self.cat)
        #self.tiger = torch.from_numpy(self.tiger)
        exp_folder = os_path.join(pyinstaller_root, "./data/exp")
        self.exp1 = np.load(os_path.join(exp_folder, "exp1.npy"))[0]
        self.exp2 = np.load(os_path.join(exp_folder, "exp2.npy"))[0]
        #self.exp3 = np.load(os_path.join(exp_folder, "exp3.npy"))
        #self.exp1 = torch.from_numpy(self.exp1)
        #self.exp2 = torch.from_numpy(self.exp2)
        vec_folder = os_path.join(pyinstaller_root, "./data")
        init_img1 = os_path.join(vec_folder, "1.png")
        init_img2 = os_path.join(vec_folder, "2.jpg")
        self.init_img1 = Image.open(init_img1)
        self.init_img2 = Image.open(init_img2)

        #self.obj1 = Face()
        #self.obj1.img2face(self.init_img1)
        #left_image = self.obj1.rendered_img
        #left_image = self.convert_cv_qt(left_image)
        #left_image = left_image.scaled(img_size, img_size, Qt.KeepAspectRatio)
        #self.image1.setPixmap(left_image)

        #self.obj2 = Face()
        #self.obj2.img2face(self.init_img2)
        #right_image = self.obj2.rendered_img
        #right_image = self.convert_cv_qt(right_image)
        #right_image = right_image.scaled(img_size, img_size, Qt.KeepAspectRatio)
        #self.image2.setPixmap(right_image)
        
        #QPixmap.fromImage()

        self.mouse_timer = QTimer(self)
        self.mouse_timer.timeout.connect(self.onTimer)
        self.mouse_timer.start(10000)
        self.setMouseTracking(True)
        self.stop_auto = False

        self.reset()

    def changeButtonState(self, button, color):
        button.setStyleSheet("background-color :" + color)
    def updateButtonState(self):
        tl = [self.tex_button_l, self.tex1_button_l, self.tex2_button_l]
        el = [self.exp_button_l, self.exp1_button_l, self.exp2_button_l]
        tr = [self.tex_button_r, self.tex1_button_r, self.tex2_button_r]
        er = [self.exp_button_r, self.exp1_button_r, self.exp2_button_r]
        for i, v in enumerate(tl):
            if i != self.tex_state_l - 1:
                self.changeButtonState(v, "#e1e1e1")
            else:
                self.changeButtonState(v, "#aeadac")
        for i, v in enumerate(el):
            if i != self.exp_state_l - 1:
                self.changeButtonState(v, "#e1e1e1")
            else:
                self.changeButtonState(v, "#aeadac")
        for i, v in enumerate(tr):
            if i != self.tex_state_r - 1:
                self.changeButtonState(v, "#e1e1e1")
            else:
                self.changeButtonState(v, "#aeadac")
        for i, v in enumerate(er):
            if i != self.exp_state_r - 1:
                self.changeButtonState(v, "#e1e1e1")
            else:
                self.changeButtonState(v, "#aeadac")

 

    def paintEvent(self, event):
        qpainter = QPainter()
        qpainter.begin(self)
        qpainter.drawPixmap(self.rect(), self.background);
        qpainter.end()

        #palette = QPalette()
        #palette.setBrush(self.backgroundRole(), QBrush(self.background))
        #self.setPalette(palette)
        ##a.setMask(pixmap.mask())
        #self.setAutoFillBackground(True)

    def set_image1(self, image):
        image = self.convert_cv_qt(image)
        image = image.scaled(img_size, img_size, Qt.KeepAspectRatio)
        self.image1.setPixmap(image)

    def set_image2(self, image):
        image = self.convert_cv_qt(image)
        image = image.scaled(img_size, img_size, Qt.KeepAspectRatio)
        self.image2.setPixmap(image)
        
    def onTimer(self):
        
        self.mouse_timer.start(40)

        tex_func_l = [lambda: self.change_tex(self.obj1.human_tex, 1, render = False), lambda: self.change_tex(self.cat, 2, render = False), lambda: self.change_tex(self.tiger, 3, render = False)]
        exp_func_l = [lambda: self.change_exp(self.obj1.org_exp, 1, render = False), lambda: self.change_exp(self.exp1, 2, render = False), lambda: self.change_exp(self.exp2, 3, render = False)]
        tex_func_r = [lambda: self.change_tex(self.obj2.human_tex, 1, left = False, render = False), lambda: self.change_tex(self.cat, 2, left = False, render = False), lambda: self.change_tex(self.tiger, 3, left = False, render = False)]
        exp_func_r = [lambda: self.change_exp(self.obj2.org_exp, 1, left = False, render = False), lambda: self.change_exp(self.exp1, 2, left = False, render = False), lambda: self.change_exp(self.exp2, 3, left = False, render = False)]

        slider1_max = self.slider1.maximum()
        slider1_min = self.slider1.minimum()
        slider2_max = self.slider2.maximum()
        slider2_min = self.slider2.minimum()
        slider3_max = self.slider3.maximum()
        slider3_min = self.slider3.minimum()
        slider1_v = self.slider1.value()
        slider2_v = self.slider2.value()
        slider3_v = self.slider3.value()

        if slider1_v == slider1_min:
            self.s1_direction = 1 # increasing direction
        elif slider1_v == slider1_max:
            self.s1_direction = -1 # decreasing direction

        if slider2_v == slider2_min:
            self.s2_direction = 1 # increasing direction
            ind = self.exp_state_r % len(exp_func_r)
            if (ind == self.exp_state_l % len(exp_func_l)):
                ind = ((ind + 1) % len(exp_func_r))
            exp_func_r[ind]()
        elif slider2_v == slider2_max:
            self.s2_direction = -1 # decreasing direction
            ind = self.exp_state_l % len(exp_func_l)
            if (ind == self.exp_state_r % len(exp_func_r)):
                ind = ((ind + 1) % len(exp_func_l))
            exp_func_l[ind]()

        if slider3_v == slider3_min:
            self.s3_direction = 1 # increasing direction
            ind = self.tex_state_r % len(tex_func_r)
            if (ind == self.tex_state_l % len(tex_func_l)):
                ind = ((ind + 1) % len(tex_func_r))
            tex_func_r[ind]()
        elif slider3_v == slider3_max:
            self.s3_direction = -1 # decreasing direction
            ind = self.tex_state_l % len(tex_func_l)
            if (ind == self.tex_state_r % len(tex_func_r)):
                ind = ((ind + 1) % len(tex_func_l))
            tex_func_l[ind]()

        self.slider1.setValue(slider1_v + self.s1_direction)
        self.slider2.setValue(slider2_v + self.s2_direction)
        self.slider3.setValue(slider3_v + self.s3_direction)

        # Rotation

        ratio = self.h_slider_image1.value() / self.h_slider_image1.maximum()
        h_angle = (ratio - 0.5) / 0.5
        ratio = self.v_slider_image1.value() / self.v_slider_image1.maximum()
        v_angle = (ratio - 0.5) / 0.5
        if h_angle == v_angle == 0:
            h_angle = 0.01
            v_angle = 0.01
        current = h_angle + 1j * v_angle

        #epsilon = 0.9
        t = 0.2
        R = t * (1 / np.abs(current)) + (1-t) * 0.4
        rot = R * np.exp(1j * 0.01 * 2 * np.pi)

        rot_result = current * rot
        h_angle = np.real(rot_result)
        v_angle = np.imag(rot_result)
        self.h_slider_image1.setValue(int(((h_angle * 0.5) + 0.5) * self.h_slider_image1.maximum()))
        self.v_slider_image1.setValue(int(((v_angle * 0.5) + 0.5) * self.v_slider_image1.maximum()))

        self.update_left_face()
        self.update_right_face()
        
    def eventFilter(self, obj, event):
        event_list = [QEvent.MouseMove, QEvent.MouseButtonPress, QEvent.MouseButtonRelease]
        if event.type() in event_list:
            self.mouse_timer.start(10000)
        return QObject.eventFilter(self, obj, event)
    def mouseMoveEvent(self, event):
        self.mouse_timer.start(10000)
    def mousePressEvent(self, event):
        self.mouse_timer.start(10000)
    def mouseReleaseEvent(self, event):
        self.mouse_timer.start(10000)
    def wheelEvent(self, event):
        self.mouse_timer.start(10000)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    def capture_image_l(self):
        if self.b_cap_l_status == "pause": #"繼續": #"Continue":
            if self.thread.Continue():
                self.b_cap_l.setText("一，二，三！") #"Continue")
                self.b_cap_l_status = "continue"#"拍摄" #"Capture"
                self.mouse_timer.stop()
        else:
            if self.thread.Pause(left = 1):
                self.b_cap_l.setText("拍攝") #Capture")
                self.b_cap_l_status = "pause" #"Continue"
                self.mouse_timer.start(10000)


    def capture_image_r(self):
        if self.b_cap_r_status == "pause": #"繼續": #"Continue":
            if self.thread.Continue():
                self.b_cap_r.setText("一，二，三！") #"Continue")
                self.b_cap_r_status = "continue"#"拍摄" #"Capture"
        else:
            if self.thread.Pause(left = 0):
                self.b_cap_r.setText("拍攝") #Capture")
                self.b_cap_r_status = "pause" #"Continue"

    def update_left_face(self):
        ratio = self.h_slider_image1.value() / self.h_slider_image1.maximum()
        h_angle = (ratio - 0.5) / 0.5
        ratio = self.v_slider_image1.value() / self.v_slider_image1.maximum()
        v_angle = (ratio - 0.5) / 0.5
        shape_ratio = self.slider1.value() / self.slider1.maximum()
        exp_ratio = self.slider2.value() / self.slider2.maximum()
        tex_ratio = self.slider3.value() / self.slider3.maximum()

        self.obj1.update_multiple_params(h_angle = h_angle, v_angle = v_angle, shape_ratio = shape_ratio, exp_ratio = exp_ratio, tex_ratio = tex_ratio, obj2 = self.obj2)
        self.set_image1(self.obj1.rendered_img)

    def update_right_face(self):
        ratio = self.h_slider_image2.value() / self.h_slider_image2.maximum()
        h_angle = (ratio - 0.5) / 0.5
        ratio = self.v_slider_image2.value() / self.v_slider_image2.maximum()
        v_angle = (ratio - 0.5) / 0.5
        self.obj2.update_multiple_params(h_angle = h_angle, v_angle = v_angle)
        self.set_image2(self.obj2.rendered_img)

    def image1_h_rot(self):
        ratio = self.h_slider_image1.value() / self.h_slider_image1.maximum()
        angle = (ratio - 0.5) / 0.5
        rendered_img = self.obj1.horizontal_rot(angle)
        self.set_image1(rendered_img)

    def image1_v_rot(self):
        ratio = self.v_slider_image1.value() / self.v_slider_image1.maximum()
        angle = (ratio - 0.5) / 0.5
        rendered_img = self.obj1.vertical_rot(angle)
        self.set_image1(rendered_img)

    def image2_h_rot(self):
        ratio = self.h_slider_image2.value() / self.h_slider_image2.maximum()
        angle = (ratio - 0.5) / 0.5
        rendered_img = self.obj2.horizontal_rot(angle)
        self.set_image2(rendered_img)

    def image2_v_rot(self):
        ratio = self.v_slider_image2.value() / self.v_slider_image2.maximum()
        angle = (ratio - 0.5) / 0.5
        rendered_img = self.obj2.vertical_rot(angle)
        self.set_image2(rendered_img)

    def slider1_func(self):
        # shape vector interpolation
        ratio = self.slider1.value() / self.slider1.maximum()
        self.obj1.interp_shape(self.obj2, ratio)
        rendered_img = self.obj1.rendered_img
        self.set_image1(rendered_img)

    def slider2_func(self):
        # expression vector interpolation
        ratio = self.slider2.value() / self.slider2.maximum()
        self.obj1.interp_exp(self.obj2, ratio)
        rendered_img = self.obj1.rendered_img
        self.set_image1(rendered_img)

    def slider3_func(self):
        # texture vector interpolation
        ratio = self.slider3.value() / self.slider3.maximum()
        self.obj1.interp_tex(self.obj2, ratio)
        rendered_img = self.obj1.rendered_img
        self.set_image1(rendered_img)


    @pyqtSlot(np.ndarray)
    def change_left_image(self, cv_img):
        self.obj1, result = self.to_3D_shape(cv_img)
        self.image_label.setPixmap(result)
        self.reset_slider()

        self.tex_state_l = 1
        self.exp_state_l = 1
        self.updateButtonState()

    @pyqtSlot(np.ndarray)
    def change_right_image(self, cv_img):
        self.obj2, result = self.to_3D_shape(cv_img)
        self.image_label.setPixmap(result)
        self.reset_slider()

        self.tex_state_r = 1
        self.exp_state_r = 1
        self.updateButtonState()

    def to_3D_shape(self, cv_img):
        """Updates the image_label with a new opencv image"""
        #self.vec = Run()
        rgb_image = cvtColor(cv_img, COLOR_BGR2RGB)
        img = Image.fromarray(np.uint8(rgb_image), mode="RGB")

        obj = Face()
        result = obj.img2face(img)
        h, w, _ = result.shape
        h, w = int(2*h), int(2*w)

        #pred_human = self.convert_cv_qt(obj.rendered_img)
        #pred_human = pred_human.scaled(img_size, img_size, Qt.KeepAspectRatio)
        result = self.convert_cv_qt(result)
        result = result.scaled(h, w, Qt.KeepAspectRatio)
        return obj, result#, pred_human

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        h, w, _ = cv_img.shape
        h, w = int(2*h), int(2*w)
        qt_img = self.convert_cv_qt(cv_img)
        qt_img = qt_img.scaled(h, w, Qt.KeepAspectRatio)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cvtColor(cv_img, COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format#.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def reset(self):
        self.thread.init()
        self.b_cap_l.setText("拍攝") #Capture")
        self.b_cap_r.setText("拍攝") #Capture")
        self.b_cap_l_status = "pause" #"Continue"
        self.b_cap_r_status = "pause" #"Continue"
        self.mouse_timer.start(10000)

        self.obj1 = Face()
        self.obj1.img2face(self.init_img1)

        self.obj2 = Face()
        self.obj2.img2face(self.init_img2)

        self.reset_slider()

        self.tex_state_l = 1
        self.exp_state_l = 1
        self.tex_state_r = 1
        self.exp_state_r = 1
        self.updateButtonState()

    def swap(self):
        self.reset_slider()
        temp_obj = self.obj1
        self.obj1 = self.obj2
        self.obj2 = temp_obj
        self.obj1.reset_current_state()
        self.obj2.reset_current_state()

        self.obj2.update_rendered_img(update_norm = True)

        self.set_image1(self.obj1.rendered_img)
        self.set_image2(self.obj2.rendered_img)

        temp = self.tex_state_l
        self.tex_state_l = self.tex_state_r
        self.tex_state_r = temp
        temp = self.exp_state_l
        self.exp_state_l = self.exp_state_r
        self.exp_state_r = temp
        self.updateButtonState()


    def reset_slider(self):
        self.h_slider_image1.setValue(int(self.h_slider_image1.maximum()/2))
        self.v_slider_image1.setValue(int(self.v_slider_image1.maximum()/2))
        self.h_slider_image2.setValue(int(self.h_slider_image2.maximum()/2))
        self.v_slider_image2.setValue(int(self.v_slider_image2.maximum()/2))
    
        self.slider1.setValue(0)
        self.slider2.setValue(0)
        self.slider3.setValue(0)

        self.update_left_face()
        self.update_right_face()

    def set_tex(self, tex, left = True, render = True):
        if left:
            self.obj1.tex = tex
            if render:
                ratio = self.slider3.value() / self.slider3.maximum()
                self.obj1.interp_tex(self.obj2, ratio)
                self.set_image1(self.obj1.rendered_img)
        else:
            self.obj2.current_face_tex = self.obj2.tex = tex
            if render:
                self.obj2.update_rendered_img()
                self.set_image2(self.obj2.rendered_img)

                ratio = self.slider3.value() / self.slider3.maximum()
                self.obj1.interp_tex(self.obj2, ratio)
                self.set_image1(self.obj1.rendered_img)

    def set_exp(self, exp, left = True, render = True):
        if left:
            self.obj1.exp = exp
            if render:
                ratio = self.slider2.value() / self.slider2.maximum()
                self.obj1.interp_exp(self.obj2, ratio)
                self.set_image1(self.obj1.rendered_img)
        else:
            self.obj2.current_face_exp = self.obj2.exp = exp
            if render:
                self.obj2.update_rendered_img(update_norm = True)
                self.set_image2(self.obj2.rendered_img)

                ratio = self.slider2.value() / self.slider2.maximum()
                self.obj1.interp_exp(self.obj2, ratio)
                self.set_image1(self.obj1.rendered_img)

    def change_tex(self, tex, state, left = True, render = True):
        self.set_tex(tex, left = left, render = render)
        if left:
            self.tex_state_l = state
        else:
            self.tex_state_r = state
        self.updateButtonState()

    def change_exp(self, exp, state, left = True, render = True):
        self.set_exp(exp, left = left, render = render)
        if left:
            self.exp_state_l = state
        else:
            self.exp_state_r = state
        self.updateButtonState()

if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    a.showMaximized()
    sys.exit(app.exec_())
