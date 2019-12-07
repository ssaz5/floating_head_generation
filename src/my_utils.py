import os
import PIL.Image as Image
import PIL.ImageFilter as ImageFilter
from PIL import  ImageChops
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import scipy as sp




def get_list_of_files(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_list_of_files(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles



def trim(im, th1=3, th2=11):
    data = np.array(im.convert('HSV'))
#     data = np.array(im_[:,:,1])
    mask = np.logical_or(data[...,1]>th1,data[...,2] <(255-th2))
    mask = Image.fromarray(255*mask.astype('uint8'))
    mask1 = mask.filter(ImageFilter.FIND_EDGES)
    mask2 = ImageChops.subtract(mask,mask1)
    mask1 = mask2.filter(ImageFilter.FIND_EDGES)
    mask2 = ImageChops.subtract(mask2,mask1)

#     mask2=mask
    
    
#     im.putalpha(mask)
    return mask2
    

def crop_image(face, face_mask,tol=0):
    # img is 2D image data
    # tol  is tolerance
    face_mask = np.array(face_mask)
    face = np.array(face)
    mask = face_mask>tol
    mask_idx = np.ix_(mask.any(1),mask.any(0))
    face_mask = Image.fromarray(face_mask[mask_idx])
    face = Image.fromarray(face[mask_idx])
    
    return face, face_mask



def resize_with_pad(im, desired_size):
    old_size = im.size
    
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    
    im = im.resize(new_size, Image.ANTIALIAS)
    
    new_im = Image.new("RGB", (desired_size, desired_size))
    
    new_im.paste(im, ((desired_size-new_size[0])//2,
                    (desired_size-new_size[1])//2))
    
    return np.array(new_im)


def gkern(kernlen_x=21,kernlen_y=21, nsig_x=3, nsig_y=4):
    """Returns a 2D Gaussian kernel array."""

    interval_x = (2*nsig_x+1.)/(kernlen_x)
    interval_y = (2*nsig_y+1.)/(kernlen_y)
    x = np.linspace(-nsig_x-interval_x/2., nsig_x+interval_x/2., kernlen_x+1)
    y = np.linspace(-nsig_y-interval_y/2., nsig_y+interval_y/2., kernlen_y+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.diff(st.norm.cdf(y))
    kernel_raw = np.sqrt(np.outer(kern1d, kern2d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3   
        
        
def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.wmin, box1.wmax], [box2.wmin, box2.wmax])
    intersect_h = _interval_overlap([box1.hmin, box1.hmax], [box2.hmin, box2.hmax])  
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1.hmax-box1.hmin, box1.wmax-box1.wmin
    w2, h2 = box2.hmax-box2.hmin, box2.wmax-box2.wmin
    
    union = w1*h1 + w2*h2 - intersect
    
    return float(intersect) / union

def get_filename(path):
    path = path.split('/')
    return path[-1]


class BoundBox:
    def __init__(self, hmin, wmin, hmax, wmax): # h- height, w - width
        self.hmin = hmin
        self.wmin = wmin
        self.hmax = hmax
        self.wmax = wmax
        self.box = [self.hmin,self.wmin]
        self.size = [int(hmax - hmin), int(wmax - wmin)]
        self.box = self.box+self.size
        
    def __repr__(self):
        return str(self.box) #  top, left, bottom, right
    
    def __getitem__(self, i):
        return self.box[i]
        
        
class Background:
    def __init__(self, path):
        
        self.image = Image.open(path)
        self.image = self.image.resize((768,768))
        
        
        
        self.face_ids = []
        self.face_boxes = []
        self.overlap_threshold = 0.1
        self.num_faces = 0
        self.name = get_filename(path)
        self.std_step = 1
        
        
        
    def place_face(self, face_path, blur=False, kernal_weights = (2., 2.5), ):
        face_img = Image.open(face_path)
        face_img_mask = trim(face_img)
        
        
        if blur:
                    
            mask = gkern(face_img.size[1], face_img.size[0], *kernal_weights)
    #         mask = mask*mask
            mask = mask/np.max(mask)
            th = (np.mean(mask) - self.std_step*np.std(mask)) 
            mask[mask<th] = np.nan
    #         mask[mask<th] = 0
    #         th2 = (np.mean(mask) - 0.5*np.std(mask)) 
    #         mask[mask<th2] = np.log(mask[mask<th2])
            mask = np.log(mask)
            mask_min = np.nanmin(mask)
    #         mask = 1.45**(mask - mask_min)
            mask = (mask - mask_min)
    #         mask[mask< (np.log(th) - mask_min)] = 0
            mask = mask/np.nanmax(mask)
#             mask = mask*768
            mask = mask*512
            mask = np.nan_to_num(mask)
            mask[mask>255] = 255

            mask_img = Image.fromarray(np.uint8(mask))
            mask_img = mask_img.filter(ImageFilter.MaxFilter(3))
            face_img_mask = mask_img
        
        face_img, face_img_mask  = crop_image(face_img, face_img_mask)
        
        
        back_img = self.image
        
        
        
        
        bbox = self.new_bbox(face_img)
        
        if self.face_boxes:
            overlap_attempts = 0
            while any(np.array([bbox_iou(bbox, face) for face in self.face_boxes]) > self.overlap_threshold):
                bbox = self.new_bbox(face_img)
                
                if overlap_attempts > 25:
                    return
        
        self.face_ids.append(get_filename(face_path))
        self.face_boxes.append(bbox)
        
        
        face_img = face_img.resize(bbox.size)
        face_img_mask = face_img_mask.resize(bbox.size, Image.ANTIALIAS)
        
        back_img.paste(face_img, (bbox.hmin, bbox.wmin), face_img_mask)
        


        self.num_faces += 1
        
        

    def new_bbox(self, face_img):
        back_img = self.image
        scaling_factor = np.min([back_img.size[0] / face_img.size[0], back_img.size[1] / face_img.size[1]])
        scale =  scaling_factor* (np.random.beta(1.5, 15)*75 + 4) /100   
        back_size = back_img.size
        face_size = [int(face_img.size[0]* scale), int(face_img.size[1] * scale)]
        xmin = int(np.random.rand(1) * (back_img.size[0] - face_size[0]))
        ymin = int(np.random.rand(1) * (back_img.size[1] - face_size[1]))
        bbox = BoundBox(xmin, ymin, xmin + face_size[0], ymin + face_size[1])
        return bbox
    
    def show(self):
        plt.imshow(self.image)
        
    def save(self, out_dir='output', writer = None):
        self.image.save('./'+ out_dir + '/' + str(self.num_faces) + '_' + self.name)
        if writer:
            self.save_annotation(writer)
        
    def save_annotation(self, writer):
        writer.write(str(self.num_faces) + '_' + self.name +'\n')
        writer.write(str(self.num_faces)+ '\n')
        if self.num_faces > 0:
            for i in self.face_boxes:
                writer.write(' '.join(map(str, i))+' 0 0 0 0 0 0\n')
        else:
            writer.write('0 0 0 0 0 0 0 0 0 0\n')
            
        return 0