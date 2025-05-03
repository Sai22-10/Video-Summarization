import cv2
import numpy as np
import os
from tkinter import Tk, filedialog, Button, Label
from tkinter.messagebox import showinfo

def load_yolo():
    net = cv2.dnn.readNet("D://FINAL PROJECT VIDEO SUMMARIZATION//video-summarization//src//yolo//yolov3-tiny.cfg",
                          "D://FINAL PROJECT VIDEO SUMMARIZATION//video-summarization//src//yolo//yolov3-tiny.weights")  # Use YOLOv3-tiny for faster processing
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    classes = []
    with open("D://FINAL PROJECT VIDEO SUMMARIZATION//video-summarization//src//yolo//coco.names", "r") as f:
        classes = [line.strip() for line in f]
    return net, output_layers, classes

def process_video(input_video, output_video, frame_skip=5, resize_factor=0.5):
    net, output_layers, classes = load_yolo()
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video}")
        return

    # Get the width and height of the frames
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_factor)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_factor)
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change codec to mp4v for .mp4 files
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"Error: Could not open video writer {output_video}")
        return

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
        
        frame = cv2.resize(frame, (width, height))
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        out.write(frame)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video processing complete. Output saved to {output_video}")

def select_input_video():
    input_video_path = filedialog.askopenfilename(title="Select Input Video", filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
    if input_video_path:
        output_video_path = os.path.splitext(input_video_path)[0] + "_output.mp4"
        process_video(input_video_path, output_video_path, frame_skip=5, resize_factor=0.5)
        showinfo("Processing Complete", f"Output saved to {output_video_path}")
        display_resultant_video(output_video_path)

def display_resultant_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Resultant Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    root = Tk()
    root.title("Video Summarization")
    root.geometry("300x150")

    label = Label(root, text="Video Summarization Tool")
    label.pack(pady=10)

    upload_button = Button(root, text="Upload Video", command=select_input_video)
    upload_button.pack(pady=10)

    root.mainloop()