import torch
import numpy as np
import cv2
from time import time


class DroneDetection:
    KNOWN_DISTANCE = 76.2  # centimeter
    # width of face in the real world or Object Plane
    KNOWN_WIDTH = 14.3  # centimeter

    def focal_length(measured_distance, real_width, width_in_rf_image):
        """
        This Function Calculate the Focal Length(distance between lens to CMOS sensor), it is simple constant we can find by using
        MEASURED_DISTACE, REAL_WIDTH(Actual width of object) and WIDTH_OF_OBJECT_IN_IMAGE
        :param1 Measure_Distance(int): It is distance measured from object to the Camera while Capturing Reference image
        :param2 Real_Width(int): It is Actual width of object, in real world (like My face width is = 14.3 centimeters)
        :param3 Width_In_Image(int): It is object width in the frame /image in our case in the reference image(found by Face detector)
        :retrun focal_length(Float):"""
        focal_length_value = (width_in_rf_image * measured_distance) / real_width
        return focal_length_value

    def distance_finder(focal_length, real_face_width, face_width_in_frame):
        """
        This Function simply Estimates the distance between object and camera using arguments(focal_length, Actual_object_width, Object_width_in_the_image)
        :param1 focal_length(float): return by the focal_length_Finder function
        :param2 Real_Width(int): It is Actual width of object, in real world (like My face width is = 5.7 Inches)
        :param3 object_Width_Frame(int): width of object in the image(frame in our case, using Video feed)
        :return Distance(float) : distance Estimated
        """
        distance = (real_face_width * focal_length) / face_width_in_frame
        return distance

    def __init__(self, capture_index, model_name):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    def get_video_capture(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """

        return cv2.VideoCapture(self.capture_index)

    def load_model(self, model_name):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                    row[3] * y_shape)
                # bgr = (0, 255, 0)
                bgr = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                # cv2.putText(frame, {},".format(row[4])", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                print(row[4])
            else:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                    row[3] * y_shape)
                # bgr = (0, 255, 0)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                # print('Swimming')
                # cv2.putText(frame, 'Swimming', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        cap = self.get_video_capture()
        assert cap.isOpened()

        while True:

            ret, frame = cap.read()
            assert ret

            frame = cv2.resize(frame, (416, 416))

            start_time = time()
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)

            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)
            # print(f"Frames Per Second : {fps}")


            # cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.putText(frame, 'Drowning',(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),1)
            cv2.putText(frame, 'Swimming',(10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),1)
            cv2.imshow('YOLOv5 Detection', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()


# Create a new object and execute.
detector = DroneDetection(capture_index='/Users/vishxwas/Downloads/Test2.mp4', model_name='/Users/vishxwas/Downloads/best.pt')
detector()
# Footer
# n = input -> 0 and path