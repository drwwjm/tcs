from ultralytics import YOLO

if __name__ ==  "__main__":
    model = YOLO('yolov8m.pt')
    torch_use_cuda_dsa = 1

    results = model.train(data=r'D:\DRIVE\TUP Files\4A\Thesis\model\tcs\data.yaml', epochs=50, imgsz=640, project=r'D:\DRIVE\TUP Files\4A\Thesis\model\tcs\results', device="CPU", lr0=0.001, lrf=0.01, fliplr=0.0)