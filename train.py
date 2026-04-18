#windows environment config
#import warnings, os
#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"    
#warnings.filterwarnings('ignore')
#use if __name__ == '__main__':

from ultralytics import YOLO
model = YOLO('OShipNet.yaml')
model.train(data='MVDD.yaml',
            imgsz=640,
            epochs=300,
            batch=32,
            patience=300,
            optimizer='SGD',
            pretrained =False,
            )
model.val(data='MVDD.yaml',split='test')
