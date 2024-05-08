import cv2
import os
from ultralytics import YOLO
import streamlit as st
import numpy as np
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode, RTCConfiguration

path = ""
VIDEOS_DIR = os.path.join(path, 'video')
model_name = 'yolov8-n-100'
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


class BanknoteDetector(VideoTransformerBase):
    def __init__(self):
        super().__init__()
        self.model = YOLO(f'models/train-{model_name}/weights/best.pt')

    def transform(self, frame):
        print('hey')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model.predict(frame)
        annotated_frame = results[0].plot()
        return cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)


def main():
    st.title("Banknote Detection")

    st.write("--Use operations in the side bar")

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
                        # Get frame dimensions (height, width)
                        # annotated_frame = frame
                        # height, width = frame.shape[:2]
                        #
                        # # Calculate center coordinates
                        # center_x = int(width / 2)
                        # center_y = int(height / 2)
                        #
                        # # Define radius (adjust as needed)
                        # radius = 50

                        # Draw the circle (BGR format for OpenCV)
                        # cv2.circle(annotated_frame, (center_x, center_y), radius, (0, 0, 255), -1)
                        # Convert frame to RGB format for Streamlit display
                        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                        # Display the processed frame on Streamlit
                        # FRAME_WINDOW.image(rgb_frame, channels="RGB")  # Adjust width as needed

                        # cv2.imshow("Yolov8 Tracking", annotated_frame)
                        # Write processed frame to output video
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
        st.header("Webcam Live Feed")
        run = st.checkbox('Run')

        if run:
            webrtc_ctx = webrtc_streamer(key="example",
                                         rtc_configuration=RTC_CONFIGURATION,
                                         video_processor_factory=BanknoteDetector,
                                         mode=WebRtcMode.SENDRECV)


if __name__ == "__main__":
    main()
