import sys
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QFileDialog, QLabel
from PyQt5.QtCore import Qt
import cv2

import matplotlib.pyplot as plt
import numpy as np
import pickle

class MyWindow(QWidget):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.data = None
        self.selection = {}
        self.selection['session'] = 0
        self.selection['roi'] = 0
        self.selection['plane'] = 0
        self.init_ui()

    def init_ui(self):
        
        self.fig = Figure(figsize=(10, 10), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        self.load_button = QPushButton('Load')
        self.load_button.clicked.connect(self.load_file)
        self.save_button = QPushButton('Save')
        self.save_button.clicked.connect(self.save_file)    

        self.valid_button = QPushButton('Valid')
        self.valid_button.clicked.connect(self.label_valid)
        self.invalid_button = QPushButton('Invalid')
        self.invalid_button.clicked.connect(self.label_invalid)

        self.prevexp_button = QPushButton('<')
        self.prevexp_button.clicked.connect(self.prev_exp)
        self.nextexp_button = QPushButton('>')
        self.nextexp_button.clicked.connect(self.next_exp) 
          
        self.prevcell_button = QPushButton('<')
        self.prevcell_button.clicked.connect(self.prev_cell)
        self.nextcell_button = QPushButton('>')
        self.nextcell_button.clicked.connect(self.next_cell)

        self.prevplane_button = QPushButton('<')
        self.prevplane_button.clicked.connect(self.prev_plane)
        self.nextplane_button = QPushButton('>')
        self.nextplane_button.clicked.connect(self.next_plane)

     

        outer_layout = QVBoxLayout()
        # layout for displaying cells
        cell_display_layout = QVBoxLayout()
        cell_display_layout.addWidget(self.canvas)
        # layout for button group labels
        button_grp_layout = QGridLayout()
        button_grp_layout.addWidget(QLabel(""),0,0,1,4)
        self.label1 = QLabel('Session')
        self.label1.setAlignment(Qt.AlignCenter)        
        button_grp_layout.addWidget(self.label1,0,4,1,2)
        self.label2 = QLabel('ROI')
        self.label2.setAlignment(Qt.AlignCenter)  
        button_grp_layout.addWidget(self.label2,0,6,1,2)
        self.label3 = QLabel('Plane')
        self.label3.setAlignment(Qt.AlignCenter)  
        button_grp_layout.addWidget(self.label3,0,8,1,2)

        # layout for buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.load_button)
        buttons_layout.addWidget(self.save_button)
        buttons_layout.addWidget(self.valid_button)
        buttons_layout.addWidget(self.invalid_button)
        buttons_layout.addWidget(self.prevexp_button)
        buttons_layout.addWidget(self.nextexp_button)
        buttons_layout.addWidget(self.prevcell_button)
        buttons_layout.addWidget(self.nextcell_button)
        buttons_layout.addWidget(self.prevplane_button)
        buttons_layout.addWidget(self.nextplane_button)


        # add nested layouts to the outer layout
        outer_layout.addLayout(cell_display_layout)
        outer_layout.addLayout(button_grp_layout)
        outer_layout.addLayout(buttons_layout)

        self.setLayout(outer_layout)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Z:
            self.data['valid'][self.selection['plane']][self.selection['roi'],[self.selection['session']]] = False
            # attempt to move to the next roi
            if self.selection['session']+2 <= self.data_meta['num_sessions']:
                # move to next session
                self.next_exp()
            else:
                # attempt to move to next roi
                self.next_cell()
            
        elif event.key() == Qt.Key_M:
            self.data['valid'][self.selection['plane']][self.selection['roi'],[self.selection['session']]] = True
            # attempt to move to the next roi
            if self.selection['session']+2 <= self.data_meta['num_sessions']:
                # move to next session
                self.next_exp()
            else:
                # attempt to move to next roi
                self.next_cell()

        elif event.key() == Qt.Key_X:
            self.prev_exp()

        elif event.key() == Qt.Key_N:
            self.next_exp()   

        elif event.key() == Qt.Key_V:
            self.prev_cell()

        elif event.key() == Qt.Key_B:
            self.next_cell()  

    
    def on_canvas_click(self, event):
        # Check if the event occurred within the axes of a subplot
        for i, ax in enumerate(self.data_meta['ax']):
            if event.inaxes == ax:
                if event.button == 1:
                    self.data['valid'][self.selection['plane']][self.selection['roi'],[i]] = True
                elif event.button == 3:
                    self.data['valid'][self.selection['plane']][self.selection['roi'],[i]] = False
        self.display_data()        


    def load_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                   "Pickle Files (*.pickle)", options=options)
        if file_name:
            with open(file_name, 'rb') as f:
                self.data = pickle.load(f)
            
            self.data_meta = {}
            self.data_meta['filename'] = file_name
            self.data_meta['num_sessions'] = len(self.data['all_fov_aligned'])
            self.data_meta['num_planes'] = len(self.data['all_fov_aligned'][0])
            self.selection['session'] = 0
            self.selection['plane'] = 0
            # build array to store valid invalid classifications
            if not 'valid' in self.data:
                self.data['valid'] = {}
                for iPlane in range(self.data_meta['num_planes']):
                    self.data['valid'][iPlane] = np.full((len(self.data['all_matches'][iPlane]),self.data_meta['num_sessions']), False, dtype=bool)

            if len(self.data['all_matches'][self.selection['plane']]) == 0:
                self.selection['roi'] = -1
            else:
                self.selection['roi'] = 0
          
            self.display_data()

    def display_data(self):
        # Here we're creating a subplot to display an image.
        # You will need to adjust this to display your own data.
        # cycle through each session making a subplot and displaying the session
        self.fig.clf()
        ax = self.fig.subplots(1,self.data_meta['num_sessions'], sharex=True, sharey=True)
        self.data_meta['ax'] = ax
        if self.selection['roi'] > -1:
            # then there are rois in the plane
            for iSession in range(self.data_meta['num_sessions']):
                # cycle through each experiment
                cell_ID = self.data['all_matches'][self.selection['plane']][self.selection['roi']][iSession]
                # axes[iSession].clear()
                # axes[iSession].imshow(all_fov_aligned[iSession][iPlane], origin='upper')
                ax[iSession].imshow(self.data['all_fov_aligned'][iSession][self.selection['plane']], origin='upper', cmap='gray')
                ax[iSession].spines['top'].set_visible(True)
                ax[iSession].spines['right'].set_visible(True)
                ax[iSession].set_xticks([])
                ax[iSession].set_yticks([])
                if self.selection['session'] == iSession:
                    # draw a thick box around it as it is the active session
                    for spine in ax[iSession].spines.values():
                        if self.data['valid'][self.selection['plane']][self.selection['roi'],iSession]:
                            spine.set_edgecolor('green')
                            spine.set_linewidth(8)
                        else:
                            spine.set_edgecolor('red')
                            spine.set_linewidth(8)                            
                else:
                    # draw a thin box around it as it is the active session
                    for spine in ax[iSession].spines.values():
                        if self.data['valid'][self.selection['plane']][self.selection['roi'],iSession]:
                            spine.set_edgecolor('green')
                            spine.set_linewidth(4)
                        else:
                            spine.set_edgecolor('red')
                            spine.set_linewidth(4)  
                    
                if not np.isnan(cell_ID):
                    roi_mask = np.zeros(self.data['all_mask_composites'][0][self.selection['plane']].shape)
                    roi_mask = self.data['all_mask_composites'][iSession][self.selection['plane']] == cell_ID
                    roi_mask = roi_mask.astype('uint8') * 255
                    _, thresh = cv2.threshold(roi_mask, 127, 255, cv2.THRESH_BINARY)
                    mask_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    mask_contours = mask_contours[0]
                    xvals = np.append(mask_contours[:,0,0],mask_contours[0,0,0])
                    yvals = np.append(mask_contours[:,0,1],mask_contours[0,0,1])
                    ax[iSession].plot(xvals,yvals,'r')

            # zoom in to region of the roi in question

            ax[0].set_xlim(self.data['all_ref_roi_crop'][self.selection['plane']][self.selection['roi']]['left'],self.data['all_ref_roi_crop'][self.selection['plane']][self.selection['roi']]['right'])
            ax[0].set_ylim(self.data['all_ref_roi_crop'][self.selection['plane']][self.selection['roi']]['bottom'],self.data['all_ref_roi_crop'][self.selection['plane']][self.selection['roi']]['top'])

            # update labels of current selections
            self.label1.setText('Session (' + str(self.selection['session']) + ')')
            self.label2.setText('ROI (' + str(self.selection['roi']) + ')')
            self.label3.setText('Plane (' + str(self.selection['plane']) + ')')
        else:
            # no ROIs in plane therefore draw a message
            self.fig.clf()
            self.label2.setText('ROI (no ROIS in plane)')

        # Make sure the canvas is updated with the new image.
        self.canvas.draw()

    def save_file(self):
        match_data = self.data
        print('Saving data to ' + self.data_meta['filename'])
        with open(self.data_meta['filename'], 'wb') as pickle_out:
            pickle.dump(match_data, pickle_out)

    def label_valid(self):
        # code 
        x = 0

    def label_invalid(self):
        # code 
        x = 0

    def prev_cell(self):
        total_rois = len(self.data['all_matches'][self.selection['plane']])
        if total_rois > 0:
            self.selection['roi'] = self.selection['roi'] - 1
            if self.selection['roi'] < 0:
                # then wrap around
                self.selection['roi'] = 0
        else:
            self.selection['roi'] = -1         
        self.selection['session'] = 0
        self.display_data()

    def next_cell(self):
        total_rois = len(self.data['all_matches'][self.selection['plane']])
        if total_rois > 0:
            self.selection['roi'] = self.selection['roi'] + 1
            if self.selection['roi'] > total_rois - 1:
                # then wrap around
                self.selection['roi'] = total_rois - 1
        else:
            self.selection['roi'] = -1
        self.selection['session'] = 0
        self.display_data()

    def prev_plane(self):
        self.data_meta['num_sessions'] = len(self.data['all_fov_aligned'])
        self.data_meta['num_planes'] = len(self.data['all_fov_aligned'][0])
        self.selection['plane'] = self.selection['plane'] - 1
        if self.selection['plane'] < 0:
            # then wrap around
            self.selection['plane'] = 0
        total_rois = len(self.data['all_matches'][self.selection['plane']])
        if total_rois > 0:
            self.selection['roi'] = 0
        else:
            self.selection['roi'] = -1
        self.selection['session'] = 0
        self.display_data()

    def next_plane(self):
        self.data_meta['num_sessions'] = len(self.data['all_fov_aligned'])
        self.data_meta['num_planes'] = len(self.data['all_fov_aligned'][0])
        self.selection['plane'] = self.selection['plane'] + 1
        if self.selection['plane'] > self.data_meta['num_planes'] - 1:
            # then wrap around
            self.selection['plane'] = self.data_meta['num_planes'] - 1
        total_rois = len(self.data['all_matches'][self.selection['plane']])
        if total_rois > 0:
            self.selection['roi'] = 0
        else:
            self.selection['roi'] = -1
        self.selection['session'] = 0
        self.display_data()

    def prev_exp(self):
        self.selection['session'] = self.selection['session'] - 1 
        if self.selection['session'] < 0:
            # then wrap around
            self.selection['session'] = self.data_meta['num_sessions'] - 1
        self.display_data()

    def next_exp(self):
        self.selection['session'] = self.selection['session'] + 1 
        if self.selection['session'] > self.data_meta['num_sessions'] - 1:
            # then wrap around
            self.selection['session'] = 0
        self.display_data()



def main():
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

# # Import the Tkinter library
# import tkinter as tk

# # Create a new window
# root = tk.Tk()

# # Set the title of the window
# root.title("My First GUI")

# # Create a label widget
# label = tk.Label(root, text="Hello, World!")

# # Add the label to the window
# label.pack()

# # Start the event loop
# root.mainloop()

# import tkinter as tk
# from tkinter import filedialog
# from matplotlib.figure import Figure
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
# import pickle
# import numpy as np
# from matplotlib.patches import Rectangle

# class GUIApp:
#     def __init__(self, root):
#         self.root = root
#         self.all_fov_aligned = None
#         self.current_iExp = 0
#         self.current_iPlane = 0
#         self.current_cell = 0
#         self.good_cells = set()
#         self.bad_cells = set()

#         self.load_button = tk.Button(root, text='Load', command=self.load_file)
#         self.load_button.pack()

#         self.fig = Figure(figsize=(5, 4), dpi=100)
#         self.canvas = FigureCanvasTkAgg(self.fig, master=root)
#         self.canvas.get_tk_widget().pack()
#         self.ax = self.fig.add_subplot(111)

#         root.bind('<Left>', self.prev_image)
#         root.bind('<Right>', self.next_image)
#         root.bind('y', self.mark_good)
#         root.bind('u', self.mark_bad)

#     def load_file(self):
#         file_path = filedialog.askopenfilename(filetypes=(("Pickle files", "*.pickle"),))
#         if file_path:
#             with open(file_path, 'rb') as f:
#                 self.all_fov_aligned = pickle.load(f)
#             self.display_image()

#     def display_image(self):
#         data = self.all_fov_aligned[self.current_iExp][self.current_iPlane]
#         self.ax.imshow(data, cmap='gray')  # Assuming the data is in grayscale
#         self.canvas.draw()

#     def prev_image(self, event=None):
#         # Update current_iExp and current_iPlane for previous image and display it
#         pass

#     def next_image(self, event=None):
#         # Update current_iExp and current_iPlane for next image and display it
#         pass

#     def mark_good(self, event=None):
#         self.good_cells.add(self.current_cell)
#         self.bad_cells.discard(self.current_cell)
        
#     def mark_bad(self, event=None):
#         self.bad_cells.add(self.current_cell)
#         self.good_cells.discard(self.current_cell)

# root = tk.Tk()
# app = GUIApp(root)
# root.mainloop()