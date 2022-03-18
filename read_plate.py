
import cv2
import numpy as np
from lib_detection import load_model, detect_lp, im2single
import socket

# Load model LP detection
wpod_net_path = "wpod-net_update1.json"
wpod_net = load_model(wpod_net_path)

# Kích thước lớn nhất và nhỏ nhất của 1 chiều ảnh
Dmax = 608
Dmin = 288

# Create a client socket
clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM);
# Connect to the server
clientSocket.connect(("127.0.0.1", 9090));

def extract_plate(img_path, i) :
    # Đọc file ảnh đầu vào
    Ivehicle = cv2.imread(img_path)
    h, w, _ = Ivehicle.shape
    # Ivehicle = Ivehicle[h//2:, :]
    Ivehicle = Ivehicle[h//2:, w//3:int(w*2/3)]

    # Lấy tỷ lệ giữa W và H của ảnh và tìm ra chiều nhỏ nhất
    ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)

    try:
        _ , LpImg, lp_type = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)

        # Crop image
        cropped_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))

        width = cropped_image.shape[1]
        img = cropped_image[:, int(0.87*width):]
        out = cv2.hconcat([cropped_image, img])
        cv2.imwrite(str(i)+".jpg", out)

        # Send data to server
        data = str(i)+".jpg"
        print(">>> ", data)
        return data

    except:
        print("Không phát hiện biển số!")
        clientSocket.send("error".encode())


list_images = [
    "test/test3.jpg",
    "test/test5.jpg",
    "test/test6.jpg",
    "test/test7.jpg",
    "test/test8.jpg"]

print("===========================================")
result = ""
for i, img_path in enumerate(list_images):
    data = extract_plate(img_path, i)
    clientSocket.sendall(data.encode())
    # result = result + "," + data

# clientSocket.sendall(result.encode())
cv2.destroyAllWindows()