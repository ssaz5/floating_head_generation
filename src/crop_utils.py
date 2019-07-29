import face_recognition
import numpy as np

from PIL import Image

def distance_from_center(center, extremas, depth=3):
    sum_left = 0
    sum_right = 0
    tot_sum = 0
    avg_slope = 0
    for i in range(depth):
        sum_left += np.linalg.norm(center - extremas[i,:])
        sum_right += np.linalg.norm(center - extremas[-i-1,:])
        dist = extremas[i,:] - extremas[-i-1,:]
        tot_sum += np.linalg.norm(dist)
        avg_slope += dist[1]/(dist[0]+1e-12)
    sum_left /= depth
    sum_right /= depth
    tot_sum /= depth
    avg_slope /=depth
    direction = (sum_left - sum_right)/tot_sum
    return sum_left, sum_right, tot_sum, direction, avg_slope


def is_face_good(direction, slope):
    return (np.sum(np.abs(direction)+np.abs(slope)) < 0.3) and (np.abs(direction) < 0.175 ) and (np.abs(slope) < 0.20 )

def get_box(center, extremas, box_mult=1):
    box = []
    left, right, tot, direction, slope = distance_from_center(center[0,:], extremas)
    if is_face_good(direction, slope):
        center_box = np.zeros((2,))
        center_box[1] = int((center[0,:][1] + center[-1,:][1])/2)
        center_box[0] = int((extremas[-1,:][0] + extremas[0,:][0]) / 2)
        box.append(center_box[0]-0.75*tot*box_mult)
        box.append(center_box[1]-1.1*tot*box_mult)
        box.append(center_box[0]+0.75*tot*box_mult)
        box.append(center_box[1]+1.1*tot*box_mult)
    
    return box
       


def crop_face(image_path, out_name, box_mult=1):
    image = face_recognition.load_image_file(image_path)
    face_landmarks_list = face_recognition.face_landmarks(image)
    if face_landmarks_list:
        chin = np.array(face_landmarks_list[0]['chin'])
        nose_bridge = np.array(face_landmarks_list[0]['nose_bridge'])

        box = np.array(get_box(nose_bridge, chin, box_mult))
#         print(box, image.shape, box.shape)
        
        if len(box) > 0: 
            if any(box < 0) or box[2] > image.shape[1] or box[3] > image.shape[0]:
                return False

            img = Image.fromarray(image)
            cropped_image = img.crop(box)
            cropped_image.save(out_name)
            return True
        
    return False
