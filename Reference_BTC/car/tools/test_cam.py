import cv2


def gstreamer_pipeline(
    capture_width=640,
    capture_height=640,
    display_width=640,
    display_height=640,
    framerate=20,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

if __name__ == "__main__":

    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    assert cap.isOpened(), "Camera failed"

    while 1:
        _, frame = cap.read()
        cv2.imwrite('runs/test_cam.jpg', frame)