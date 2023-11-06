from __future__ import annotations

from PIL import Image
from ultralytics import YOLO
import cv2


def process_image(model_path: str, image_path: str) -> list:
    model = YOLO(model_path)
    results = model(image_path)  # Results list
    predictions = []

    # Show the results
    for r in results:
        predictions.append(r)
        im_array = r.plot()  # Plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.show()  # Show image

    return predictions


def process_video(model_path: str, video_path: str) -> list:
    # Load the YOLOv8 model
    model = YOLO(model_path)
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    predictions = []

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame, verbose=False)
            # Visualize the results on the frame
            predictions.append(results[0])
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

    return predictions
