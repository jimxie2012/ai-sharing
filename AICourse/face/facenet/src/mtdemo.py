import argparse,cv2
from mtcnn import MTCNN
from face_align import face_preprocess

class CCEShare:

    def __init__(self,file_name):
        self.m_mtcnn = MTCNN('./mtcnn.pb')
        self.m_raw_img = cv2.imread(file_name)

    def Detect(self):
        bbox, scores, landmarks = self.m_mtcnn.detect(self.m_raw_img)
        return bbox.astype('int32'),landmarks.astype('int32')

    def DrawBox(self,bbox):
        for box in bbox:
            self.m_raw_img = cv2.rectangle(self.m_raw_img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 30)

    def DrawLandMark(self,landmarks):
        for pts in landmarks:
            for i in range(5):
                self.m_raw_img = cv2.circle(self.m_raw_img, (pts[i+5], pts[i]), 30, (0, 255, 0), -1)

    def SaveImg(self,file_name):
        cv2.imwrite(file_name,self.m_raw_img)

def process_mtcnn(imgpath):
    share = CCEShare(imgpath)
    bbox, landmarks = share.Detect()
    share.DrawBox(bbox)
    share.SaveImg("./image/mtcnn-box.jpg")
    share.DrawLandMark(landmarks)
    share.SaveImg("./image/mtcnn-mark.jpg")

def process_align(imgpath):
    image_array = cv2.imread(imgpath)

    face,landmarks = face_preprocess(image = image_array,landmark_model_type='large',crop_size=640)
    mark_face = face.copy()
    for key in landmarks:
        for ps in landmarks[key]:
            cv2.circle(mark_face,ps,10,(0,255,0),-1)

    cv2.imwrite("./image/align-face.jpg",face)
    cv2.imwrite("./image/align-mark.jpg",mark_face)

if __name__ == '__main__':
    process_mtcnn('./image/mtcnn-0.jpg')
    process_align('./image/mtcnn-0.jpg')
