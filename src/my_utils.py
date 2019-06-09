import os
import PIL.Image as Image
import PIL.ImageFilter as ImageFilter
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
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])  
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    
    union = w1*h1 + w2*h2 - intersect
    
    return float(intersect) / union

def get_filename(path):
    path = path.split('/')
    return path[-1]


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.size = (xmax - xmin, ymax - ymin)
        
        
        
class Background:
    def __init__(self, path):
        
        self.image = Image.open(path)
        self.face_ids = []
        self.face_boxes = []
        self.overlap_threshold = 0.1
        self.num_faces = 0
        self.name = get_filename(path)
        
        
        
    def place_face(self, face_path, blur=False, kernal_weights = (2., 2.5), ):
        face_img = Image.open(face_path)
        self.face_ids.append(get_filename(face_path))
        
        back_img = self.image
        
        bbox = self.new_bbox(face_img)
        
        if self.face_boxes:
            while any(np.array([bbox_iou(bbox, face) for face in self.face_boxes]) > self.overlap_threshold):
                bbox = self.new_bbox(face_img)
        
        self.face_boxes.append(bbox)
        face_img = face_img.resize(bbox.size)
        mask = gkern(face_img.size[1], face_img.size[0], *kernal_weights)
#         mask = mask*mask
        mask = mask/np.max(mask)
        th = (np.mean(mask) - 1*np.std(mask)) 
        mask[mask<th] = np.nan
#         mask[mask<th] = 0
        if not blur:
            mask[mask>=th] = 255
        else:
#             th2 = (np.mean(mask) - 0.5*np.std(mask)) 
#             mask[mask<th2] = np.log(mask[mask<th2])
            mask = np.log(mask)
            mask_min = np.nanmin(mask)
#             mask = 1.45**(mask - mask_min)
            mask = (mask - mask_min)
#             mask[mask< (np.log(th) - mask_min)] = 0
            mask = mask/np.nanmax(mask)
            mask = mask*768
            mask[mask>255] = 255
            
        mask_img = Image.fromarray(np.uint8(mask))
        mask_img = mask_img.filter(ImageFilter.MaxFilter(3))
        back_img.paste(face_img, (bbox.xmin, bbox.ymin), mask_img)
        self.num_faces += 1
        

    def new_bbox(self, face_img):
        back_img = self.image
        scaling_factor = np.min([back_img.size[0] / face_img.size[0], back_img.size[1] / face_img.size[1]])
        scale =  (np.random.rand(1)*5 + 2) /  scaling_factor 
        back_size = back_img.size
        face_size = (face_img.size[0] // scale, face_img.size[1] // scale)
        xmin = int(np.random.rand(1) * (back_img.size[0] - face_size[0]))
        ymin = int(np.random.rand(1) * (back_img.size[1] - face_size[1]))
        bbox = BoundBox(xmin, ymin, xmin + face_size[0], ymin + face_size[1])
        return bbox
    
    def show(self):
        plt.imshow(self.image)
        
    def save(self, out_dir='output'):
        self.image.save('./'+ out_dir + '/' + str(self.num_faces) + '_' + self.name)
        