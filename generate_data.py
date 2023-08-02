import cv2
import numpy as np
import os


def combine(image):
    image = cv2.resize(image, (600, 600))
    M = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), np.random.randint(0, 180), 1.0)
    rotated_0 = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    M = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), np.random.randint(180, 360), 1.0)
    rotated_1 = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REPLICATE)

    mask_top = 1*(rotated_1[:,:,3]> 200)
    mask_bottom = 1*(rotated_0[:,:,3] > 200)
    mask_bottom = np.clip(mask_bottom - mask_top, 0, 1)
    blend = (1-(mask_top[:, :, np.newaxis]))*rotated_0[:,:,:3] + mask_top[:, :, np.newaxis]*rotated_1[:,:,:3]
    blend = cv2.GaussianBlur(np.uint8(blend),(5,5),0)
    return blend, mask_top, mask_bottom

def detect_contours(image):
    contours, __ = cv2.findContours(np.uint8(image*255),cv2.RETR_LIST , cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if len(contours) > 1:
        contours = contours[:2]
    return contours

def format_yolo(contours, identity_index, width, height):
    annotation = str()
    for _, j in enumerate(contours):
        annotation += str(identity_index)
        for coord in j:
            annotation += " {} {}".format(coord[0][0]/width, coord[0][1]/height)
        annotation += "\n"
    return annotation

def create_dataset(folder):
    try:
        os.makedirs(folder+"/val/images/")
        os.makedirs(folder+"/val/labels/")
        os.makedirs(folder+"/train/images/")
        os.makedirs(folder+"/train/labels/")
    except:
        ...

def create_data(number):
    create_dataset("test")
    image = cv2.imread("test.png", cv2.IMREAD_UNCHANGED) # Read the object with transparency
    for i in range(number):
        blended, mask_top, mask_bottom = combine(image)
        top = format_yolo(detect_contours(mask_top), 0, mask_top.shape[1], mask_top.shape[0])
        bottom = format_yolo(detect_contours(mask_bottom), 1, mask_top.shape[1], mask_top.shape[0])
        if i < 0.75*number:
            cv2.imwrite("test/train/images/{:06d}.png".format(i), blended)
            with open("test/train/labels/{:06d}.txt".format(i), "w") as f:
                f.write(top)
                f.write(bottom)
        else:
            cv2.imwrite("test/val/images/{:06d}.png".format(i), blended)
            with open("test/val/labels/{:06d}.txt".format(i), "w") as f:
                f.write(top)
                f.write(bottom)
create_data(5000)
