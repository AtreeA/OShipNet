from ultralytics import YOLO
model = YOLO('OShipNet.yaml')
model.train(data='MVDD.yaml',
            imgsz=640,
            epochs=300,
            batch=4,
            patience=300,
            optimizer='SGD',
            pretrained =False,
            )
model.val(data='MVDD.yaml',split='test')