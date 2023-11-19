# UIT Car Racing 2023
## Car
Sử dụng để chạy mô hình YOLOv5 và Segmentation trên Jetson Nano
### Car Structure
```
car
│   reset_cam.sh                  #Reset camera
│   run.py        
│
├───export
│       onnx2trt.py
│       torch2torchscript.py
│
├───images                         #Test images                      
│   ├───detect
│   │       bus.jpg
│   │       zidane.jpg
│   │
│   └───segment
│           980.jpg
│           982.jpg
│
├───lib                            #Library
│   ├───cfg
│   │       cfg.yaml               #Configuration
│   │
│   ├───control
│   │       UITCar.py
│   │
│   ├───model
│   │       trt.py
│   │       unet.py
│   │
│   └───utils
│           plots.py
│           utils.py
│
├───runs
├───tools
│       test_cam.py
│       test_torchscript.py
│       test_trt.py
│
└───weights                         #Pretrain
```
### Export model
ONNX to TRT:
```
python3 export/onnx2trt.py --onnx best_obj.onnx --engine best_obj.engine
```
Torchscript:
```
python3 export/torch2torchscript.py --weight best_seg.pt
```

### Demo Test
```
├───tools
│       test_cam.py                 
│       test_torchscript.py
│       test_trt.py
```
#### Test Segmentation model
```
python3 tools/test_torchscript.py
```
#### Test YOLOv5n model
```
python3 tools/test_trt.py
```
#### Test camera
```
python3 tools/test_cam.py
```

## Training Model
Run in Colab to training model with custom datasets
```
Training_Model
├───Segmentation
│       segmentation_training.ipynb
│
└───YOLOv5
        yolov5_training.ipynb
```
#### Segmentation model: 
[seg.pt](https://drive.google.com/file/d/1O6SlW6f3TEhCISBTDo1cRNhEijqaFfhL/view?usp=sharing)
#### Sign traffic object model:
[obj.onnx](https://drive.google.com/file/d/1Q_RJ68S8nbuH2B2J6Z7xnQ072-M6--oC/view?usp=sharing)
