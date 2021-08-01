# -*- coding:utf-8 _*-
import math
import cv2
from PIL import Image, ImageDraw
from matplotlib.pyplot import imshow
import face_recognition
from collections import defaultdict
import numpy as np

def detect_landmark(image_array, model_type="large"):
    face_landmarks_list = face_recognition.face_landmarks(image_array, model=model_type)
    face_landmarks_list = face_landmarks_list[0]
    return face_landmarks_list

def align_face(image_array, landmarks):
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    left_eye_center = np.mean(left_eye, axis=0).astype("int")
    right_eye_center = np.mean(right_eye, axis=0).astype("int")
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    angle = math.atan2(dy, dx) * 180. / math.pi
    eye_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                  (left_eye_center[1] + right_eye_center[1]) // 2)
    rotate_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
    rotated_img = cv2.warpAffine(image_array, rotate_matrix, (image_array.shape[1], image_array.shape[0]))
    return rotated_img, eye_center, angle

def rotate(origin, point, angle, row):
    x1, y1 = point
    x2, y2 = origin
    y1 = row - y1
    y2 = row - y2
    angle = math.radians(angle)
    x = x2 + math.cos(angle) * (x1 - x2) - math.sin(angle) * (y1 - y2)
    y = y2 + math.sin(angle) * (x1 - x2) + math.cos(angle) * (y1 - y2)
    y = row - y
    return int(x), int(y)

def rotate_landmarks(landmarks, eye_center, angle, row):
    rotated_landmarks = defaultdict(list)
    for facial_feature in landmarks.keys():
        for landmark in landmarks[facial_feature]:
            rotated_landmark = rotate(origin=eye_center, point=landmark, angle=angle, row=row)
            rotated_landmarks[facial_feature].append(rotated_landmark)
    return rotated_landmarks

def corp_face(image_array, size, landmarks):
    x_min = np.min(landmarks['chin'], axis=0)[0]
    x_max = np.max(landmarks['chin'], axis=0)[0]
    x_center = (x_max - x_min) / 2 + x_min
    left, right = (x_center - size / 2, x_center + size / 2)

    eye_landmark = landmarks['left_eye'] + landmarks['right_eye']
    eye_center = np.mean(eye_landmark, axis=0).astype("int")
    lip_landmark = landmarks['top_lip'] + landmarks['bottom+lip']
    lip_center = np.mean(lip_landmark, axis=0).astype("int")
    mid_part = lip_center[1] - eye_center[1]
    top, bottom = eye_center[1] - (size - mid_part) / 2, lip_center[1] + (size - mid_part) / 2

    pil_img = Image.fromarray(image_array)
    left, top, right, bottom = [int(i) for i in [left, top, right, bottom]]
    cropped_img = pil_img.crop((left, top, right, bottom))
    cropped_img = np.array(cropped_img)
    return cropped_img, left, top

def transfer_landmark(landmarks, left, top):
    transferred_landmarks = defaultdict(list)
    for facial_feature in landmarks.keys():
        for landmark in landmarks[facial_feature]:
            transferred_landmark = (landmark[0] - left, landmark[1] - top)
            transferred_landmarks[facial_feature].append(transferred_landmark)
    return transferred_landmarks

def face_preprocess(image, landmark_model_type='large', crop_size=140):
    face_landmarks_dict = detect_landmark(image_array=image, model_type=landmark_model_type)
    aligned_face, eye_center, angle = align_face(image_array=image, landmarks=face_landmarks_dict)
    rotated_landmarks = rotate_landmarks(landmarks=face_landmarks_dict,
                                         eye_center=eye_center, angle=angle, row=image.shape[0])
    cropped_face, left, top = corp_face(image_array=aligned_face, size=crop_size, landmarks=rotated_landmarks)
    transferred_landmarks = transfer_landmark(landmarks=rotated_landmarks, left=left, top=top)
    return cropped_face, transferred_landmarks

if __name__=='__main__':
    # load image
    img_name = '/Users/JimXie/Desktop/what/image/mtcnn-0.jpg'
    image_array = cv2.imread(img_name)

    # preprocess the face image
    face,landmarks = face_preprocess(image = image_array,
                                     landmark_model_type='large',
                                     crop_size=640)

    print(landmarks)