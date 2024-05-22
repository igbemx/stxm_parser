import pandas as pd
from ipyfilechooser import FileChooser
import matplotlib.pyplot as plt
import os
import cv2
import h5py
import skimage
from hdr_import import HDRFile
#%matplotlib widget
import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mplcursors
import numpy as np
from scipy.ndimage import fourier_shift
from skimage.registration import phase_cross_correlation
import math
from nexusformat import nexus
from tifffile import imwrite
from skimage import exposure

class ROIs_select():
    def __init__(self,im, vmin, vmax, stack, energies, fc):
        self.im = im
        self.fc = fc
        self.energies = energies
        self.stack = stack
        self.selected_points = []
        self.fig = plt.figure(figsize = (16,8))
        self.ax2 = plt.subplot(3,2,2)
        self.ax3 = plt.subplot(3,2,4)
        self.ax4 = plt.subplot(3,2,6)
        self.ax1 = plt.subplot(1,2,1)
        self.img = self.ax1.imshow(self.im.copy(), vmin = vmin, vmax = vmax)
        self.ka = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.i_roi_button = widgets.Button(description='Intensity ROI')
        self.bg_roi_button = widgets.Button(description='Background ROI')
        self.append_i_roi = widgets.Button(description='Append I ROI')
        self.fName_input = widgets.Text(placeholder='Type spectrum name',disabled=False)
        self.save_spectrum = widgets.Button(description='Save spectrum to .txt')
        vbox1 = widgets.VBox([self.i_roi_button, self.bg_roi_button])
        vbox2 = widgets.VBox([self.fName_input, self.save_spectrum])
        hbox1 = widgets.HBox([vbox1, self.append_i_roi, vbox2])
        
        display(hbox1)
        #display(self.bg_roi_button)
        #display(self.append_i_roi)
        self.i_roi_button.on_click(self.intensity_roi)
        self.bg_roi_button.on_click(self.bg_roi)
        self.append_i_roi.on_click(self.append_roi_pts)
        self.save_spectrum.on_click(self.write_spectrum_file)
        self.i_roi_pts = []
        self.bg_roi_pts = []
        self.appended_roi = []
        self.bg_lst = []
        self.i_lst = []
        self.avr = []
        
    def poly_img(self,img,pts):
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1,1,2))
        color = (250,1,1)
        cv2.polylines(img,[pts],True, color, 1)        
        return img

    def onclick(self, event):      
        self.selected_points.append([event.xdata,event.ydata])        
        if len(self.selected_points)>1:
            self.fig            
            self.img.set_data(self.poly_img(self.im.copy(),self.selected_points))  
            self.fig.canvas.draw()
            
    
    def disconnect_mpl(self,_):
        self.fig.canvas.mpl_disconnect(self.ka)
        
    def append_roi_pts(self, _):
        self.appended_roi.append(self.selected_points)
        self.selected_points = []   
        
        
    def intensity_roi(self, _):
        self.i_lst = []
        if len(self.appended_roi) == 0:
            self.i_roi_pts = self.selected_points
            self.i_lst = []
            for img in self.stack:
                i_elem = self.i_bg_mean(img, self.i_roi_pts)
                self.i_lst.append(i_elem)
            self.ax2.clear()
            
            self.ax22 = self.ax2.twiny()                          
            self.ax22.clear()
            self.ax2.plot(self.energies,self.i_lst, label='Sample intensity')
            self.ax2.set_ylabel('Pixel value (a.u.)')
            self.ax22.set_xlim(self.ax2.get_xlim())
            length_of_E = len(np.array(self.energies))
            slice_param = 3
            for n in range(1,20):
                if int(length_of_E / n) <= 15:
                    slice_param = n
                    break     
                    
            new_E_array = self.energies[0::slice_param]            
            new_idx_array = [idx for idx, i in enumerate(self.energies) if i in new_E_array]
            self.ax22.set_xticks(new_E_array)
            self.ax22.set_xticklabels(new_idx_array)
            self.ax22.set_xlabel('Image number')
            
            self.selected_points = []
            if len(self.bg_lst) != 0 and len(self.i_lst) != 0 or len(self.bg_lst) != 0 and len(self.avr) != 0:
                self.plot_spectrum()
        else:
            self.avr = []            
            i_lst_final = []
            for elem in self.appended_roi:
                i_roi_pts = elem
                i_lst = []
                for img in self.stack:
                    i_elem = self.i_bg_mean(img, i_roi_pts)
                    i_lst.append(i_elem)
                i_lst_final.append(i_lst)
           
            self.avr = self.averager(i_lst_final)
            self.ax2.clear()
            self.ax2.plot(self.energies,self.avr, label='Sample intensity')
            self.ax2.set_ylabel('Pixel value (a.u.)')
            self.selected_points = [] 
            self.appended_roi = []
            if len(self.bg_lst) != 0 and len(self.i_lst) != 0 or len(self.bg_lst) != 0 and len(self.avr) != 0:
                self.plot_spectrum()
        self.ax2.legend()
        
    def bg_roi(self, _):
        self.bg_roi_pts = self.selected_points
        self.bg_lst = []
        for img in self.stack:
            bg_elem = self.i_bg_mean(img, self.bg_roi_pts)
            self.bg_lst.append(bg_elem)
        self.ax3.clear()
        self.ax3.plot(self.energies,self.bg_lst, label='Background') 
        self.ax3.set_ylabel('Pixel value (a.u.)')
        self.ax3.legend()
        if len(self.bg_lst) != 0 and len(self.i_lst) != 0 or len(self.bg_lst) != 0 and len(self.avr) != 0:
            self.plot_spectrum()
        
        self.selected_points = [] 
        
        
    def i_bg_mean(self, img, sel_pnts):
        arr, mask, roi = [], [], []
        arr = np.array([sel_pnts],'int')
        mask = cv2.fillPoly(np.zeros(img.shape,np.uint8),arr,[1,1,1])
        roi = np.multiply(img,mask)
        roi = np.asarray(roi, dtype='float64') 
        roi[roi == 0] = np.nan
        return np.nanmean(roi)
    
    def plot_spectrum(self):
        
        if len(self.i_lst) != 0:
            intensities = np.array(self.i_lst)
        else:
            intensities = np.array(self.avr)
        self.spectrum = np.zeros(len(intensities))
        background = np.array(self.bg_lst)
        self.spectrum = background/intensities
        self.ax4.clear()
        self.ax4.plot(self.energies,self.spectrum, label='spectrum')
        self.ax4.set_xlabel("Photon Energy (eV)")
        self.ax4.set_ylabel("Optical Density (a.u.)")
        self.ax4.legend()
        
    def averager(self, arr):
        result = []
        for i in range(len(arr[0])):
            elem_avr = 0
            for j in range(len(arr)):
                elem_avr += arr[j][i]
            elem_avr /= len(arr)
            result.append(elem_avr)
        return result   
    
    def write_spectrum_file(self, _):
        spectrum_name = str(self.fName_input.value)
        fName = self.fc.selected_path + '/' + spectrum_name + '.txt'
        with open(fName, 'w') as f:
            for i in range(len(self.energies)):
                f.write(f"      {self.energies[i]}      {self.spectrum[i]}\n")
            f.close()
class Nexus_File_Save:
    def __init__(self, img_array, sample_hdr, fc):
        self.img_array = img_array
        self.sample_hdr = sample_hdr
        self.fc = fc
        
        self.fName_input = widgets.Text(placeholder='Type file name',disabled=False)
        self.output_label = widgets.Label()
        self.save_button = widgets.Button(description='Save NEXUS file')
        self.save_tif_button = widgets.Button(description='Save ImageJ TIF')
        self.save_button.on_click(self.save_nexus_root)
        self.save_tif_button.on_click(self.save_tif)
        vbox0 = widgets.VBox([self.save_button, self.save_tif_button])
        hbox1 = widgets.HBox([self.fName_input, vbox0])
        vbox1 = widgets.VBox([hbox1, self.output_label])
        display(vbox1)
        
        # collecting the main keys from the microscope .hdr file
        scan_def = sample_hdr.as_dict['ScanDefinition']
        scan_name = scan_def['Label']
        scan_type = scan_def['Type']
        scan_dwell = scan_def['Dwell']
        scan_regions = int(list(scan_def['Regions'])[0])
        x_axis = scan_def['Regions']['1']['PAxis']
        y_axis = scan_def['Regions']['1']['QAxis']
        energy = scan_def['StackAxis']
        
        self.scan_name = scan_name
        
        #make nexus root
        self.root = nexus.NXroot()
        self.root['entry1'] = nexus.NXentry()
        self.root['entry1/ScanDefinition'] = nexus.NXentry()
        self.root['entry1/ScanDefinition/Name'] = scan_name
        self.root['entry1/ScanDefinition/Type'] = scan_type
        self.root['entry1/ScanDefinition/DwellTime'] = nexus.NXfield(scan_dwell, units='ms')
        self.root['entry1/ScanDefinition/Regions'] = scan_regions
        self.root['entry1/ScanDefinition/X_axis'] = nexus.NXentry()
        self.root['entry1/ScanDefinition/X_axis/Name'] = x_axis['Name']
        self.root['entry1/ScanDefinition/X_axis/Unit'] = x_axis['Unit']
        self.root['entry1/ScanDefinition/X_axis/Min'] = nexus.NXfield(x_axis['Min'], units='µm')
        self.root['entry1/ScanDefinition/X_axis/Max'] = nexus.NXfield(x_axis['Max'], units='µm')
        self.root['entry1/ScanDefinition/X_axis/Points'] = x_axis['Points'][0]
        self.root['entry1/ScanDefinition/Y_axis'] = nexus.NXentry()
        self.root['entry1/ScanDefinition/Y_axis/Name'] = y_axis['Name']
        self.root['entry1/ScanDefinition/Y_axis/Unit'] = y_axis['Unit']
        self.root['entry1/ScanDefinition/Y_axis/Min'] = nexus.NXfield(y_axis['Min'], units='µm')
        self.root['entry1/ScanDefinition/Y_axis/Max'] = nexus.NXfield(y_axis['Max'], units='µm')
        self.root['entry1/ScanDefinition/Y_axis/Points'] = y_axis['Points'][0]
        self.root['entry1/ScanDefinition/Energy'] = nexus.NXentry()
        self.root['entry1/ScanDefinition/Energy/Unit'] = energy['Unit']
        self.root['entry1/ScanDefinition/Energy/Min'] = nexus.NXfield(energy['Min'], units='eV')
        self.root['entry1/ScanDefinition/Energy/Max'] = nexus.NXfield(energy['Max'], units='eV')
        self.root['entry1/ScanDefinition/Energy/Points'] = energy['Points'][0]
        self.root['entry1/Collection'] = nexus.NXentry()
        self.root['entry1/counter0'] = nexus.NXdata()
        array_shape = (len(self.img_array), len(self.img_array[0][0]), len(self.img_array[0]))
        self.root['entry1/counter0/data'] = nexus.NXfield(self.img_array, shape=array_shape, dtype=np.float64)
        self.root['entry1/counter0/sample_x'] = nexus.NXfield(x_axis['Points'][1:], dtype=np.float64)
        self.root['entry1/counter0/sample_y'] = nexus.NXfield(y_axis['Points'][1:], dtype=np.float64)
        self.root['entry1/counter0/energy'] = nexus.NXfield(energy['Points'][1:], dtype=np.float64)
        
        dicty = self.sample_hdr.as_dict
        
        for key in dicty.keys():
            self.root['entry1/Collection/'+str(key)] = nexus.NXentry()
            if type(dicty[key]) == dict:
                for key1 in dicty[key].keys():
                    self.root['entry1/Collection/'+str(key)+'/'+str(key1)] = nexus.NXentry()
                    if type(dicty[key][key1]) == dict:
                        for key2 in dicty[key][key1].keys():
                            self.root['entry1/Collection/'+str(key)+'/'+str(key1)+'/'+str(key2)] = nexus.NXentry()
                            try:
                                if type(dicty[key][key1][key2]) == dict:
                                    for key3 in dicty[key][key1][key2].keys():
                                        self.root['entry1/Collection/'+str(key)+'/'+str(key1)+'/'+str(key2)+'/'+str(key3)] = dicty[key][key1][key2][key3]
                                else:
                                    self.root['entry1/Collection/'+str(key)+'/'+str(key1)+'/'+str(key2)] = dicty[key][key1][key2]
                            except:
                                pass
                    else:
                        self.root['entry1/Collection/'+str(key)+'/'+str(key1)] = dicty[key][key1]
            else:
                self.root['entry1/Collection/'+str(key)] = dicty[key]
                
    def save_nexus_root(self, _):
        if len(self.fName_input.value) == 0:
            self.output_label.value = 'The data was saved with its default name. You can also provide a custom name...'
            file_name = self.scan_name.split('.')[0]
        else:
            file_name = str(self.fName_input.value) + '.hdf5'
            
        path = self.fc.selected_path + '/'
        fileName = path + file_name
        self.root.save(filename=fileName)   
        
    def save_tif(self, _):
        if len(self.fName_input.value) == 0:
            self.output_label.value = 'The data was saved with its default name. You can also provide a custom name...'
            file_name = self.scan_name.split('.')[0] + '.tif'
        else:
            file_name = str(self.fName_input.value) + '.tif'
            
        path = self.fc.selected_path + '/'
        fname = path + file_name
        im_array = np.asarray(self.img_array, dtype='uint16')
        imwrite(fname, im_array)

class line_intensity():

    def __init__(self,im, extent):
        self.im = im
        self.extent = extent
        self.selected_points = []
        self.fig,self.ax = plt.subplots(1,2, figsize=(16,8))
        self.img = self.ax[0].imshow(self.im.copy())
        self.ax[1].plot(np.zeros(30))
        self.ax[1].set_ylabel('Intensity (a.u.)')
        self.ka = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        #disconnect_button = widgets.Button(description="Disconnect mpl")
        self.plot_line_intensity = widgets.Button(description='Plot Line Intensity')
        #Disp.display(disconnect_button)
        display(self.plot_line_intensity)
        self.plot_line_intensity.on_click(self.plot_intensity)
        #disconnect_button.on_click(self.disconnect_mpl)
        
    def line_img(self,img,pts):
        pts = np.array(pts, np.int32)
        #pts = pts.reshape((-1,1,2))
        start_p = pts[0]
        end_p = pts[1]
        cv2.line(img, (start_p), (end_p), (2500,250,250), 1, cv2.LINE_AA)
        return img

    def onclick(self, event):
        if len(self.selected_points) < 2:
            self.selected_points.append([event.xdata,event.ydata])
            if len(self.selected_points) == 2:
                self.fig
                self.img.set_data(self.line_img(self.im.copy(),self.selected_points))
    '''def disconnect_mpl(self,_):
        self.fig.canvas.mpl_disconnect(self.ka)'''
        
    def plot_intensity(self, _):
        img_px_x = len(self.im[0])
        img_px_y = len(self.im)
        img_size_x = np.abs(self.extent[1] - self.extent[0])
        img_size_y = np.abs(self.extent[3] - self.extent[2])
        start_stop = self.selected_points
        start_x = int(start_stop[0][0])
        stop_x = int(start_stop[1][0])
        start_y = int(start_stop[0][1])
        stop_y = int(start_stop[1][1])
        start_real_x = np.abs(start_x * img_size_x / img_px_x - self.extent[1])
        stop_real_x = np.abs(stop_x * img_size_x / img_px_x - self.extent[1])
        start_real_y = np.abs(start_y * img_size_y / img_px_y - self.extent[3])
        stop_real_y = np.abs(stop_y * img_size_y / img_px_y - self.extent[3])
        
        rr,cc = skimage.draw.line(start_x, start_y, stop_x, stop_y)
        diff_x = np.abs(rr[-1] - rr[0])
        diff_y = np.abs(cc[-1] - cc[0])
        ax_x = np.linspace(start_real_x, stop_real_x, len(rr))
        ax_y = np.linspace(start_real_y, stop_real_y, len(cc))
        self.ax[1].clear() 
        self.ax[1].set_ylabel('Intensity (a.u.)')
        if diff_y > diff_x:            
            self.ax[1].plot(ax_y, self.im[cc, rr])
            self.ax[1].set_xlabel('Sample Y (µm)')
        else:
            self.ax[1].plot(ax_x, self.im[cc, rr])
            self.ax[1].set_xlabel('Sample X (µm)')
        self.selected_points = []

def remove_hot_dead_pixels(data,tolerance=3,worry_about_edges=True):    
    from scipy.ndimage import median_filter
    Z = data
    for i in range(len(Z)):
        for j in range(len(Z[i])):
            try:
                if Z[i][j] == 0:
                    Z[i][j] = 1
            except:
                print(f"Couldn't find {i} or {j}")
    #print(f"Data is {Z}")
    blurred = median_filter(Z, size=2)
    #print(f"Blurred is: {blurred}")
    difference = Z - blurred
    #print(f"Difference is: {difference}")
    threshold = 5*np.std(difference)
    #print(f"Threshold is: {threshold}")
    threshold_d = (-5)*np.std(difference)
    #print(f"Threshold_d is: {threshold_d}")
    
    #find the hot pixels, but ignore the edges
    hot_pixels = np.nonzero((np.abs(difference[1:-1,1:-1])>threshold) )
    #print(f"Hot_pixels1 is: {hot_pixels}")
    hot_pixels = np.array(hot_pixels) + 1 #because we ignored the first row and first column
    #print(f"Hot_pixels2 is: {hot_pixels}")
    fixed_image = np.copy(data) #This is the image with the hot pixels removed
    for y,x in zip(hot_pixels[0],hot_pixels[1]):
        fixed_image[y,x]=blurred[y,x]
        
    dead_pixels = np.nonzero((np.abs(difference[1:-1,1:-1])<threshold_d) )
    #print(f"dead_pixels1 is: {dead_pixels}")
    dead_pixels = np.array(dead_pixels) + 1 #because we ignored the first row and first column
    #print(f"dead_pixels2 is: {dead_pixels}")
    #fixed_image = np.copy(data) #This is the image with the hot pixels removed
    for y,x in zip(dead_pixels[0],dead_pixels[1]):
        fixed_image[y,x]=blurred[y,x]
    
    if worry_about_edges == True:
        height,width = np.shape(data)

        ###Now get the pixels on the edges (but not the corners)###

        #left and right sides
        for index in range(1,height-1):
            #left side:
            med  = np.median(data[index-1:index+2,0:2])
            diff = np.abs(data[index,0] - med)
            if diff>threshold: 
                hot_pixels = np.hstack(( hot_pixels, [[index],[0]]  ))
                fixed_image[index,0] = med
            elif diff<threshold_d: 
                dead_pixels = np.hstack(( dead_pixels, [[index],[0]]  ))
                fixed_image[index,0] = med

            #right side:
            med  = np.median(data[index-1:index+2,-2:])
            diff = np.abs(data[index,-1] - med)
            if diff>threshold: 
                hot_pixels = np.hstack(( hot_pixels, [[index],[width-1]]  ))
                fixed_image[index,-1] = med
            elif diff<threshold_d: 
                dead_pixels = np.hstack(( dead_pixels, [[index],[width-1]]  ))
                fixed_image[index,-1] = med

        #Then the top and bottom
        for index in range(1,width-1):
            #bottom:
            med  = np.median(data[0:2,index-1:index+2])
            diff = np.abs(data[0,index] - med)
            if diff>threshold: 
                hot_pixels = np.hstack(( hot_pixels, [[0],[index]]  ))
                fixed_image[0,index] = med
            elif diff<threshold_d: 
                dead_pixels = np.hstack(( dead_pixels, [[0],[index]]  ))
                fixed_image[0,index] = med

            #top:
            med  = np.median(data[-2:,index-1:index+2])
            diff = np.abs(data[-1,index] - med)
            if diff>threshold: 
                hot_pixels = np.hstack(( hot_pixels, [[height-1],[index]]  ))
                fixed_image[-1,index] = med
            elif diff<threshold_d: 
                dead_pixels = np.hstack(( dead_pixels, [[height-1],[index]]  ))
                fixed_image[-1,index] = med

        ###Then the corners###

        #bottom left
        med  = np.median(data[0:2,0:2])
        diff = np.abs(data[0,0] - med)
        if diff>threshold: 
            hot_pixels = np.hstack(( hot_pixels, [[0],[0]]  ))
            fixed_image[0,0] = med
        elif diff<threshold_d: 
            dead_pixels = np.hstack(( dead_pixels, [[0],[0]]  ))
            fixed_image[0,0] = med

        #bottom right
        med  = np.median(data[0:2,-2:])
        diff = np.abs(data[0,-1] - med)
        if diff>threshold: 
            hot_pixels = np.hstack(( hot_pixels, [[0],[width-1]]  ))
            fixed_image[0,-1] = med
        elif diff<threshold_d: 
            dead_pixels = np.hstack(( dead_pixels, [[0],[width-1]]  ))
            fixed_image[0,-1] = med

        #top left
        med  = np.median(data[-2:,0:2])
        diff = np.abs(data[-1,0] - med)
        if diff>threshold: 
            hot_pixels = np.hstack(( hot_pixels, [[height-1],[0]]  ))
            fixed_image[-1,0] = med
        elif diff<threshold_d: 
            dead_pixels = np.hstack(( dead_pixels, [[height-1],[0]]  ))
            fixed_image[-1,0] = med

        #top right
        med  = np.median(data[-2:,-2:])
        diff = np.abs(data[-1,-1] - med)
        if diff>threshold: 
            hot_pixels = np.hstack(( hot_pixels, [[height-1],[width-1]]  ))
            fixed_image[-1,-1] = med
        elif diff<threshold_d: 
            dead_pixels = np.hstack(( dead_pixels, [[height-1],[width-1]]  ))
            fixed_image[-1,-1] = med

    return hot_pixels,dead_pixels,fixed_image

def clean_images(img_array):
    clean_img_array = []
    for i in range(len(img_array)):
        img = img_array[i]
        hot_pixels,dead_pixels,fixed_image = remove_hot_dead_pixels(img)
        clean_img_array.append(fixed_image) 
    return np.array(clean_img_array)
    
    
    
    
def align_images(images, sequential, shift_param, sample_factor):
    aligned_img_array = []
    aligned_img_array_3 = []
    aligned_img_array_2 = []
    #cleaned_img_array = []
    ime = np.array(images[0], copy=True)
    im  = exposure.equalize_adapthist(ime)
    aligned_img_array_2.append(im)
    aligned_img_array.append(images[0])
    shift = [1,1]
    iter_counter = 0
    images = np.array(images)
    
    for i in range(len(images)):
        im1 = np.array(images[i], copy=True)
        im2 = exposure.equalize_adapthist(im1)
        aligned_img_array_3.append(im2)
        
    

    for i in range(len(images)-1):
            img_11 = np.array(images[0], copy=True)
            img_1 = np.array(aligned_img_array_3[0], copy=True)
            img_22 = np.array(images[i+1], copy=True)
            img_2 = np.array(aligned_img_array_3[i+1], copy=True)
            shift, error, diffphase = phase_cross_correlation(img_11, img_22, upsample_factor=sample_factor)
            offset_img_2 = fourier_shift(np.fft.fftn(img_22), shift)
            offset_img_2 = np.real(np.fft.ifftn(offset_img_2))
            aligned_img_array.append(offset_img_2)
            offset_img_22 = fourier_shift(np.fft.fftn(img_2), shift)
            offset_img_22 = np.real(np.fft.ifftn(offset_img_22))
            aligned_img_array_2.append(offset_img_22)
    
    
    while True:
        iter_counter += 1
        shift_1, shift_2 = [], []
        for i in range(len(aligned_img_array)-1):
            if sequential:
                img_1 = aligned_img_array[i]
                img_11 = aligned_img_array_2[i]
                
            else:
                img_1 = images[0]
                img_11 = aligned_img_array_2[0]
            img_2 = aligned_img_array[i+1]
            img_22 = aligned_img_array_2[i+1]
            
            #img_22 = aligned_img_array[i+1]
            shift, error, diffphase = phase_cross_correlation(img_11, img_22, upsample_factor=sample_factor)
            if shift[0] <= 5 or shift[1] <= 5:
                offset_img_2 = fourier_shift(np.fft.fftn(img_2), shift)
                offset_img_2 = np.real(np.fft.ifftn(offset_img_2))
                aligned_img_array[i+1] = offset_img_2
                
                offset_img_22 = fourier_shift(np.fft.fftn(img_22), shift)
                offset_img_22 = np.real(np.fft.ifftn(offset_img_22))
                aligned_img_array_2[i+1] = offset_img_22
                shift_1.append(shift[0])
                shift_2.append(shift[1])
            else:
                shift = [0, 0]
                img_1 = aligned_img_array[0]
                img_2 = aligned_img_array[i+1]
                shift, error, diffphase = phase_cross_correlation(img_11, img_22, upsample_factor=sample_factor)
                offset_img_2 = fourier_shift(np.fft.fftn(img_2), shift)
                offset_img_2 = np.real(np.fft.ifftn(offset_img_2))
                aligned_img_array[i+1] = offset_img_2
                offset_img_22 = fourier_shift(np.fft.fftn(img_22), shift)
                offset_img_22 = np.real(np.fft.ifftn(offset_img_22))
                aligned_img_array_2[i+1] = offset_img_22
                shift_1.append(shift[0])
                shift_2.append(shift[1])
        max_shift_1 = abs(max(shift_1))
        min_shift_1 = abs(min(shift_1))
        max_shift_2 = abs(max(shift_2))
        min_shift_2 = abs(min(shift_2))
        print(max_shift_1, max_shift_2, min_shift_1, min_shift_2)
        #shift_param = 0.0025
        if max_shift_1 < shift_param and max_shift_2 < shift_param and min_shift_1 < shift_param and min_shift_2 < shift_param:
            print(f"Done after {iter_counter} iterations")
            break
        elif iter_counter >= 20:
            print(f"Force stopped!")
            break
    return np.asarray(aligned_img_array)
            
def plot_stack(stack, extent):
    
    plt.figure(figsize=(6, 6))
    num_of_images = len(stack)
    @widgets.interact(im_number=(0, num_of_images - 1, 1))
    def update(im_number = 0):
        """Showing the image."""
        plt.imshow(stack[(im_number)], extent=extent)
        #, norm=LogNorm()
        
def make_array_stack(images):
    stack_array = []
    for i in range(len(images.xim)):
        img = images.xim[i]
        stack_array.append(np.flipud(img))
    stack_array = np.asarray(stack_array)
    return stack_array

def import_images(fc):
    file_path = fc.selected_path
    sample_hdr = HDRFile(file_path)
    #print("Length of sample_hdr", len(sample_hdr.xim))
    num_of_images = len(sample_hdr.xim)
    min_x = float(sample_hdr.as_dict['ScanDefinition']['Regions']['1']['PAxis']['Min'])
    max_x = float(sample_hdr.as_dict['ScanDefinition']['Regions']['1']['PAxis']['Max'])
    min_y = float(sample_hdr.as_dict['ScanDefinition']['Regions']['1']['QAxis']['Min'])
    max_y = float(sample_hdr.as_dict['ScanDefinition']['Regions']['1']['QAxis']['Max'])
    energy = np.array(sample_hdr.as_dict['ScanDefinition']['StackAxis']['Points'][1:])
    
    extent = [min_x, max_x, min_y, max_y]
    stack = make_array_stack(sample_hdr)
    #print("Length of stack initially: ", len(stack))
    sorting_tool = sample_hdr.sort_tool
    stack1 = [x for _, x in sorted(zip(sorting_tool, stack))]
    energy = energy[:len(stack1)]
    return stack1, extent, energy, sample_hdr

class bbox_select():

    def __init__(self,im):
        self.im = im
        self.selected_points = []
        self.fig,ax = plt.subplots(figsize=(10,10))
        self.img = ax.imshow(self.im.copy())
        self.ka = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.i_roi_button = widgets.Button(description='Intensity ROI')
        self.bg_roi_button = widgets.Button(description='Background ROI')
        display(self.i_roi_button)
        display(self.bg_roi_button)
        self.i_roi_button.on_click(self.intensity_roi)
        self.bg_roi_button.on_click(self.bg_roi)
        
        self.i_roi_pts = []
        self.bg_roi_pts = []
        
    def poly_img(self,img,pts):
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img,[pts],True,(204,102,0),2)        
        return img

    def onclick(self, event):
        #display(str(event))        
        self.selected_points.append([event.xdata,event.ydata])        
        if len(self.selected_points)>1:
            self.fig            
            self.img.set_data(self.poly_img(self.im.copy(),self.selected_points))  
            self.fig.canvas.draw()
            
    def disconnect_mpl(self,_):
        self.fig.canvas.mpl_disconnect(self.ka)
    
    def intensity_roi(self, _):
        self.i_roi_pts = self.selected_points
        #print(self.i_roi_pts)
        self.selected_points = []        
        
    def bg_roi(self, _):
        self.bg_roi_pts = self.selected_points
        #print(self.bg_roi_pts)
        self.selected_points = [] 
        
def i_bg_mean(img, sel_pnts):
    arr, mask, roi = [], [], []
    arr = np.array([sel_pnts],'int')
    mask = cv2.fillPoly(np.zeros(img.shape,np.uint8),arr,[1,1,1])
    roi = np.multiply(img,mask)
    roi = np.asarray(roi, dtype='float64') 
    roi[roi == 0] = np.nan
    return np.nanmean(roi)

def plot_stack_roi(stack):
    
    #fig, ax = plt.subplots(figsize=(6, 6))
    num_of_images = len(stack)
    @widgets.interact(im_number=(0, num_of_images - 1, 1))
    def update(im_number = 0):
        """Showing the image."""
        roi_select = bbox_select(stack[(im_number)])
        
def get_roi_intensities(stack, sel_pnts_i, sel_pnts_bg):
    i_lst, bg_lst = [], []
    for img in stack:
        i_elem = i_bg_mean(img, sel_pnts_i)
        bg_elem = i_bg_mean(img, sel_pnts_bg)
        i_lst.append(i_elem)
        bg_lst.append(bg_elem)
    return i_lst, bg_lst