from ultralytics import YOLO

# last letter determines model size (nano,small,medium,large,x-large)
# larger models are slower but more accurate
model = YOLO('yolov8n.pt')

# main loop
if __name__ == '__main__':
    while True:
        result = model.predict(source="0", show=True)
