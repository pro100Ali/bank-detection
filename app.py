import cv2
import os
from ultralytics import YOLO
import streamlit as st
import numpy as np

path = ""
VIDEOS_DIR = os.path.join(path, 'video')
model_name = 'yolov8-n-100'


def main():
    st.title("Banknote Detection")

    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("**Select an option ** ")
    st.sidebar.write("")

    activities = [
        "Video(Video detection)", "Camera(live detection)"]
    choice = st.sidebar.selectbox("select an option", activities)

    if choice == "Video(Video detection)":
        video_file = st.file_uploader(
            "Upload Video", type=['avi', 'mp4', 'mov'])
        if video_file:
            with open(os.path.join(VIDEOS_DIR, video_file.name), "wb") as f:
                f.write(video_file.getbuffer())

            video_path_out = '{}_{}_out.mp4'.format(video_file.name, model_name)
            if os.path.exists(os.path.join(VIDEOS_DIR, video_file.name)):
                cap = cv2.VideoCapture(os.path.join(VIDEOS_DIR, video_file.name))
                if not cap.isOpened():
                    raise IOError("Unable to open video file.")

                # Get video properties
                ret, frame = cap.read()
                if not ret:
                    raise ValueError("Unable to read video frame.")
                H, W, _ = frame.shape
                out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'h264'), int(cap.get(cv2.CAP_PROP_FPS)),
                                      (W, H))
                if not out.isOpened():
                    raise IOError("Unable to create video writer.")
                model_path = os.path.join('.', 'runs', 'detect', 'train19', 'weights', 'best.pt')
                # model = YOLO(model_path)
                model = YOLO(f'models/train-{model_name}/weights/best.pt')

                while cap.isOpened():
                    success, frame = cap.read()

                    if success:
                        results = model.predict(frame)
                        annotated_frame = results[0].plot()
                        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        out.write(annotated_frame)
                        # if cv2.waitKey(1) & 0xFF == ord("q"):
                        #     break

                    else:
                        break

                # Release resources
                cap.release()
                out.release()
                # cv2.destroyAllWindows()
                with open(video_path_out, "rb") as video_file:
                    video_bytes = video_file.read()
                st.video(video_bytes)
    elif choice == "Camera(live detection)":
        st.title("Webcam Live Feed")
        run = st.checkbox('Run')
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)
        model = YOLO(f'models/train-{model_name}/weights/best.pt')

        while run:
            _, frame = camera.read()

            results = model.predict(frame)
            annotated_frame = results[0].plot()
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(annotated_frame_rgb)


if __name__ == "__main__":
    main()
