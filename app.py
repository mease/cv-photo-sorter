###############################################
# Derek Mease
# CSCI 4831-5722 (Fleming)
# Final Project - Computer Vision Photo Sorter
###############################################

# This file contains code for the GUI application to allow viewing of
# photos as well as sorting and filtering.

import sys
import textwrap
import pickle
import warnings
from os import listdir
from os.path import isfile, join, exists
from PyQt5.QtCore import Qt, QSize, QByteArray, QDataStream, QIODevice
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImageReader, QDoubleValidator, QImage
import cv2 as cv
import similarity
import cascades
import dnn

warnings.filterwarnings("ignore", category=DeprecationWarning)

IMAGE_DIR = './photos/'

# Checks filenames for proper image extensions.
def filename_has_image_extension(filename):
    valid_img_extensions = \
        ['bmp', 'gif', 'jpg', 'jpeg', 'png', 'pbm', 'pgm', 'ppm', 'xbm', 'xpm']
    filename = filename.lower()
    extension = filename[-3:]
    four_char = filename[-4:]  # exclusively for jpeg
    if extension in valid_img_extensions or four_char in valid_img_extensions:
        return True
    else:
        return False


# Dialog for selecting options.
class OptionsDialog(QDialog):
    def __init__(self, sift_thresh=0.05, orb_thresh=0.05):
        QDialog.__init__(self)
        self.setWindowTitle('Options')
        self.setWindowModality(Qt.ApplicationModal)

        layout = QVBoxLayout(self)

        # Option for the SIFT sorting threshold
        sift_label = QLabel()
        sift_label.setText('SIFT threshold:')
        self.sift_thresh = QLineEdit(self)
        self.sift_thresh.setValidator(QDoubleValidator())
        self.sift_thresh.setText(str(sift_thresh))
        sift_layout = QHBoxLayout()
        sift_layout.addWidget(sift_label)
        sift_layout.addWidget(self.sift_thresh)
        layout.addLayout(sift_layout)

        # Option for the ORB sorting threshold
        orb_label = QLabel()
        orb_label.setText('ORB threshold:')
        self.orb_thresh = QLineEdit(self)
        self.orb_thresh.setValidator(QDoubleValidator())
        self.orb_thresh.setText(str(orb_thresh))
        orb_layout = QHBoxLayout()
        orb_layout.addWidget(orb_label)
        orb_layout.addWidget(self.orb_thresh)
        layout.addLayout(orb_layout)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_orb_thresh(self):
        return float(self.orb_thresh.text())

    def get_sift_thresh(self):
        return float(self.sift_thresh.text())


# General-purpose progress bar
class ProgressWidget(QDialog):
    def __init__(self, total, title='Loading...'):
        QWidget.__init__(self)

        self.value = 0
        self.setWindowTitle(title)
        self.setWindowModality(Qt.ApplicationModal)
        self.pbar = QProgressBar(self)
        self.pbar.setMaximum(total)
        self.pbar.setValue(0)
        self.pbar.setGeometry(50, 20, 300, 20)
        self.status = QLabel(self)
        self.status.setMaximumWidth(300)
        self.status.setAlignment(Qt.AlignLeft)
        self.status.setGeometry(50, 50, 300, 60)
        self.pbar.show()

    def increment(self, n):
        self.value += n
        self.pbar.setValue(self.value)
        QApplication.processEvents()

    def set_value(self, n):
        self.value = n
        self.pbar.setValue(self.value)
        QApplication.processEvents()

    def set_status(self, msg, wrap=True):
        if wrap:
            msg = textwrap.fill(msg, width=50)
        self.status.setText(msg)
        QApplication.processEvents()


# Displays the image for the selected thumbnail
class DisplayImage(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self)
        self.parent = parent
        self.pixmap = QPixmap()
        self.label = QLabel(self)
        self.assigned_img_full_path = ''
        self.image = None

    # Update the image in the main display
    def update_display_image(self, path_to_image='', dnn_detection=None, cascade_detection=None):
        self.assigned_img_full_path = path_to_image
        self.image = cv.imread(path_to_image)
        
        # Draw detections from DNN
        if dnn_detection is not None:
            self.image = dnn.draw_boxes(self.image,
                                        dnn_detection.boxes,
                                        dnn_detection.confidences,
                                        dnn_detection.idxs,
                                        dnn_detection.label)
        
        # Draw dtections from HAAR cascades
        if cascade_detection is not None:
            self.image = cascades.draw_boxes(self.image, cascade_detection)

        # render the display image when a thumbnail is selected
        self.on_main_window_resize()

    # Rescale image on window resize
    def on_main_window_resize(self, event=None):
        main_window_size = self.parent.size()
        main_window_height = main_window_size.height()
        main_window_width = main_window_size.width()

        display_image_max_height = main_window_height - 50
        display_image_max_width = main_window_width - 200

        height, width, channel = self.image.shape
        bytesPerLine = 3 * width
        qImg = QImage(self.image.data, width, height,
                      bytesPerLine, QImage.Format_RGB888).rgbSwapped()

        self.pixmap = QPixmap.fromImage(qImg)
        self.pixmap = self.pixmap.scaled(
            QSize(display_image_max_width, display_image_max_height),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation)

        self.label.setPixmap(self.pixmap)


# A group of images for display in the ImageFileSelector
class ImageGroup(QWidget):
    def __init__(self, files, progress, album_path='', display_image=None, parent=None, bgcolor='', dnn_detections=None, cascade_detections=None):
        QWidget.__init__(self, parent=parent)
        self.display_image = display_image
        self.parent = parent
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(f'background-color: {bgcolor}')
        self.grid_layout = QGridLayout(self)
        self.grid_layout.setVerticalSpacing(30)
        self.dnn_detections = dnn_detections
        self.cascade_detections = cascade_detections

        row_in_grid_layout = 0
        first_img_file_path = ''

        # Load thumbnails from cache for faster loading
        thumbnails = {}
        thumbnail_cache = './data/thumbnails.pickle'
        if exists(thumbnail_cache):
            with open(thumbnail_cache, 'rb') as f:
                state = pickle.load(f)
            
            for (key, buffer) in state:
                qpixmap = QPixmap()
                stream = QDataStream(buffer, QIODevice.ReadOnly)
                stream >> qpixmap
                thumbnails[key] = qpixmap

        for file_name in files:
            if filename_has_image_extension(file_name) is False:
                continue
            img_label = QLabel()
            text_label = QLabel()
            img_label.setAlignment(Qt.AlignCenter)
            text_label.setAlignment(Qt.AlignCenter)
            file_path = join(album_path, file_name)
            progress.set_status(f'Loading {file_path}')
            
            if file_name in thumbnails.keys():
                img_label.setPixmap(thumbnails[file_name])
            else:
                imageReader = QImageReader(file_path)
                imageReader.setAutoTransform(True)
                pixmap = QPixmap.fromImageReader(imageReader)
                pixmap = pixmap.scaled(
                    QSize(100, 100), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                thumbnails[file_name] = pixmap
                img_label.setPixmap(pixmap)

            text_label.setText(textwrap.fill(file_name, width=15))
            text_label.setMaximumWidth(100)
            img_label.mousePressEvent = lambda e, index = row_in_grid_layout, file_path = file_path: self.on_thumbnail_click(
                e, index, file_path)
            text_label.mousePressEvent = img_label.mousePressEvent
            thumbnail = QBoxLayout(QBoxLayout.TopToBottom)
            thumbnail.addWidget(img_label)
            thumbnail.addWidget(text_label)
            self.grid_layout.addLayout(
                thumbnail, row_in_grid_layout, 0, Qt.AlignCenter)

            if row_in_grid_layout == 0:
                first_img_file_path = file_path
            row_in_grid_layout += 1
            progress.increment(1)

        # Cache thumbnails
        state = []
        for key, value in thumbnails.items():
            qbyte_array = QByteArray()
            stream = QDataStream(qbyte_array, QIODevice.WriteOnly)
            stream << value
            state.append((key, qbyte_array))
        with open(thumbnail_cache, 'wb') as f:
            pickle.dump(state, f)

    # Update the main image display when thumbnail is clicked
    def on_thumbnail_click(self, event, index, img_file_path):
        if not isfile(img_file_path):
            return

        self.parent.clear_selection()

        # Select the single clicked thumbnail
        text_label_of_thumbnail = self.grid_layout.itemAtPosition(
            index, 0).itemAt(1).widget()
        text_label_of_thumbnail.setStyleSheet("background-color:lightblue;")

        # Update the display image
        if self.dnn_detections is not None:
            self.display_image.update_display_image(
                img_file_path, self.dnn_detections[index], None)
        elif self.cascade_detections is not None:
            self.display_image.update_display_image(
                img_file_path, None, self.cascade_detections[index])
        else:
            self.display_image.update_display_image(img_file_path)


# Image file selector widget
class ImageFileSelector(QWidget):
    def __init__(self, parent=None, album_path='', display_image=None, sort=None, thresh=None, filter=None, filter_cascade=None):
        QWidget.__init__(self, parent=parent)
        self.album_path = album_path
        self.display_image = display_image
        self.grid_layout = QGridLayout(self)
        self.grid_layout.setVerticalSpacing(30)

        # Get all the image files in the directory
        files = [f for f in listdir(self.album_path)
                 if isfile(join(self.album_path, f))]
        if len(files) == 0:
            return

        row_in_grid_layout = 0
        first_img_file_path = ''

        # Sort images into groups using histogram, SIFT, or ORB features
        if sort is not None:
            if sort == 'hist':
                sorter = similarity.HistogramGrouper(self.album_path)
            elif sort == 'sift':
                sorter = similarity.Sift(
                    self.album_path, similarity_threshold=thresh)
            elif sort == 'orb':
                sorter = similarity.Orb(
                    self.album_path, similarity_threshold=thresh)
            else:
                raise 'Invalid sorting choice: ' + sort

            if (sort == 'sift' or sort == 'orb'):
                progress = ProgressWidget(len(files), f'Calculating {sort.upper()} features...')
                progress.setGeometry(100, 100, 400, 130)
                progress.show()
                sorter.get_all_features(progress, files)
                progress.close()

            progress = ProgressWidget(len(files), 'Grouping images...')
            progress.setGeometry(100, 100, 400, 130)
            progress.show()
            group_list = sorter.group_similar(progress, files)
            progress.close()
        else:
            group_list = [files]
        
        # Filter images classified with HAAR cascades
        cascade_detections = None
        if filter_cascade is not None:
            progress = ProgressWidget(len(files), 'Filtering images...')
            progress.setGeometry(100, 100, 400, 130)
            progress.show()

            filtered, cascade_detections = cascades.filter(self.album_path, files, filter_cascade, progress)
            group_list = [filtered]

            progress.close()

        # Filter images classified with DNN
        dnn_detections = None
        if filter is not None:
            progress = ProgressWidget(len(files), 'Filtering images...')
            progress.setGeometry(100, 100, 400, 130)
            progress.show()

            yolo = dnn.Yolo()
            filtered, dnn_detections = yolo.filter(
                self.album_path, files, filter, progress)
            group_list = [filtered]

            progress.close()

        num_imgs_in_groups = len([item for sublist in group_list for item in sublist])
        progress = ProgressWidget(num_imgs_in_groups, 'Loading images...')
        progress.setGeometry(100, 100, 400, 130)
        progress.show()

        # Load images groups into the selector
        self.groups = []
        for g in group_list:
            if len(g) > 0:
                bgcolor = ['coral', 'thistle'][row_in_grid_layout % 2]
                group = ImageGroup(g, progress, album_path,
                                   display_image, self, bgcolor, dnn_detections, cascade_detections)
                self.groups.append(group)
                self.grid_layout.addWidget(
                    group, row_in_grid_layout, 0, Qt.AlignCenter)
                if row_in_grid_layout == 0:
                    first_img_file_path = join(self.album_path, g[0])
                row_in_grid_layout += 1

        progress.close()

        # Automatically select the first file in the list during init
        if len(self.groups) > 0:
            self.groups[0].on_thumbnail_click(None, 0, first_img_file_path)

    def clear_selection(self):
        # Deselect all thumbnails in the image selector
        for group in self.groups:
            for text_label_index in range(len(group.grid_layout)):
                text_label = group.grid_layout.itemAtPosition(
                    text_label_index, 0).itemAt(1).widget()
                text_label.setStyleSheet("background-color:none;")


# Main application widget
class MainWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.options = OptionsDialog()
        self.options.setGeometry(100, 100, 200, 130)

        self.resizeEvent = lambda e: self.on_main_window_resize(e)

        self.display_image = DisplayImage(self)
        self.image_file_selector = ImageFileSelector(
            album_path=IMAGE_DIR,
            display_image=self.display_image)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(175)
        nav = scroll
        nav.setWidget(self.image_file_selector)

        self.layout = QGridLayout(self)
        self.layout.addWidget(nav, 0, 0, Qt.AlignLeft)
        self.layout.addWidget(self.display_image.label, 0, 1)
        self.setLayout(self.layout)

    # Refresh the image selector
    def refresh_nav(self, sort=None, thresh=None, filter=None, filter_cascade=None):
        self.image_file_selector = ImageFileSelector(
            album_path=IMAGE_DIR,
            display_image=self.display_image, sort=sort, thresh=thresh, filter=filter, filter_cascade=filter_cascade)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(175)
        nav = scroll
        nav.setWidget(self.image_file_selector)
        self.layout.itemAtPosition(0, 0).widget().close()
        self.layout.addWidget(nav, 0, 0, Qt.AlignLeft)

    # Show the options dialog
    def show_options(self):
        self.options.show()

    # Sort by histogram
    def hist_sort(self):
        self.refresh_nav('hist', None, None, None)

    # Sort by SIFT features
    def sift_sort(self):
        thresh = self.options.get_sift_thresh()
        self.refresh_nav('sift', thresh, None, None)

    # Sort by ORB features
    def orb_sort(self):
        thresh = self.options.get_orb_thresh()
        self.refresh_nav('orb', thresh, None, None)

    # Filter by DNN classification
    def filter(self, label):
        self.refresh_nav(None, None, label, None)

    # Filter by HAAR cascade classification
    def filter_cascade(self, label):
        self.refresh_nav(None, None, None, label)

    # Resize displayed image on window resize
    def on_main_window_resize(self, event):
        self.display_image.on_main_window_resize(event)


# Main application
class App(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set main window attributes
        self.title = 'Computer Vision - Photo Sorter'
        self.left = 50
        self.top = 50
        self.width = 800
        self.height = 600

        self.main_widget = MainWidget()
        self.setCentralWidget(self.main_widget)
        self.createMenuBar()

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.show()

    # Create the top menu bar
    def createMenuBar(self):
        menuBar = QMenuBar(self)

        optionsAct = QAction('&Options', self)
        optionsAct.setStatusTip('Options')
        optionsAct.triggered.connect(self.main_widget.show_options)

        refreshAct = QAction('&Refresh', self)
        refreshAct.setStatusTip('Refresh images')
        refreshAct.triggered.connect(self.main_widget.refresh_nav)

        exitAct = QAction('&Exit', self)
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(qApp.quit)

        histSortAct = QAction('&Histogram', self)
        histSortAct.setStatusTip('Group by similar histograms')
        histSortAct.triggered.connect(self.main_widget.hist_sort)

        siftSortAct = QAction('&SIFT', self)
        siftSortAct.setStatusTip('Group by similar SIFT features')
        siftSortAct.triggered.connect(self.main_widget.sift_sort)

        orbSortAct = QAction('&ORB', self)
        orbSortAct.setStatusTip('Group by similar ORB features')
        orbSortAct.triggered.connect(self.main_widget.orb_sort)

        fileMenu = menuBar.addMenu('&File')
        fileMenu.addAction(optionsAct)
        fileMenu.addAction(refreshAct)
        fileMenu.addAction(exitAct)

        sortMenu = menuBar.addMenu("&Sort")
        sortMenu.addAction(histSortAct)
        sortMenu.addAction(siftSortAct)
        sortMenu.addAction(orbSortAct)

        filterMenu = menuBar.addMenu("&Filter")

        filterCascadeMenu = filterMenu.addMenu("&HAAR Cascade")
        for label in ['car', 'person', 'face', 'cat']:
            filtAct = QAction(f'&{label}', self)
            filtAct.setStatusTip(f'Filter {label}')
            filtAct.triggered.connect(
                lambda state, x=label: self.main_widget.filter_cascade(x))
            filterCascadeMenu.addAction(filtAct)

        filterDNNMenu = filterMenu.addMenu("&DNN")
        for label in dnn.Yolo().labels:
            filtAct = QAction(f'&{label}', self)
            filtAct.setStatusTip(f'Filter {label}')
            filtAct.triggered.connect(
                lambda state, x=label: self.main_widget.filter(x))
            filterDNNMenu.addAction(filtAct)

        self.setMenuBar(menuBar)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
