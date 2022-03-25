
import numpy as np
import cv2
import argparse

img_path = "/home/burak/catkin_ws/src/yolo_try/src/CIMG0112.JPG"
video_path = 'WhatsApp Video 2018-05-25 at 02.51.26.mp4'


def start_webcam():
    cap = cv2.VideoCapture(0)
    return cap

#### Load Yolo #################

def load_yolo():
    net = cv2.dnn.readNetFromDarknet("/home/burak/catkin_ws/src/yolo_try/src/yolov3.cfg" , "/home/burak/catkin_ws/src/yolo_try/src/yolov3.weights")
    
    classes = []
    with open("/home/burak/catkin_ws/src/yolo_try/src/coco.names","r") as f:
        classes = [line.strip() for line in f.readlines()]

    #layer_names = net.getLayerNames()
    #output_layers = [layer_names for layer_names in net.getUnconnectedOutLayersNames()] ---> It can be used above opencv 3.4.2 or the lasts version
    layers = net.getLayerNames()
    output_layers = [layers[i[0]-1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0,255, size=(len(classes),3))
    return  classes,output_layers,colors,net



def load_image(img_path):

    img = cv2.imread(img_path)    
    img = cv2.resize(img, None, fx=0.3, fy=0.25)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)

    height,width,channels = img.shape

    return height,width,channels,img

def detect_objec(net,img,output_layers):

    blob = cv2.dnn.blobFromImage(img,scalefactor=0.00392, size=(320,320),mean=(0,0,0),swapRB=True,crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    return blob,outputs

def get_box_dimention(outputs,height,width):

    boxes = []
    confs = []
    class_ids = []

    for output in outputs:
        for detect in output:
            scores = detect[5:]
            #print(scores)
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                center_x = int(detect[0]*width)
                center_y = int(detect[1]*height)
                w = int(detect[2]*width)
                h = int(detect[3]*height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x,y,w,h])
                confs.append(float(conf))
                class_ids.append(class_id)


    return boxes,class_ids,confs


def draw_labels(boxes,class_ids,confs,classes,img,colors):

    indexes = cv2.dnn.NMSBoxes(boxes,confs,0.5,0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
            cv2.putText(img,label, (x,y-5),font, 1,color,1)
    cv2.imshow('img',img)
    


def image_detect(img_path):

    classes, output_layers, colors, model = load_yolo()
    height, width, channels, img = load_image(img_path)
    blob,outputs = detect_objec(model,img,output_layers)
    boxes, class_ids, confs = get_box_dimention(outputs,height,width)
    draw_labels(boxes,class_ids,confs,classes,img,colors)

    while  True:
        key = cv2.waitKey(1)
        if key == 27:
            break


def webcam_detect():
    classes, output_layers, colors, model = load_yolo()
    cap = start_webcam()
    while True:
        _, frame = cap.read()
        height,width,channels = frame.shape
        blob,outputs = detect_objec(model,frame,output_layers)
        boxes,class_ids,confs = get_box_dimention(outputs,height,width)
        draw_labels(boxes,class_ids,confs,classes,frame,colors)


def start_video():
    classes, output_layers, colors, model = load_yolo()
    cap = cv2.VideoCapture(video_path)
    while True:
        _,frame = cap.read()
        height,width,channels = frame.shape
        blob,outputs = detect_objec(model,frame,output_layers)
        boxes,class_ids,confs = get_box_dimention(outputs,height,width)
        draw_labels(boxes,class_ids,confs,classes,frame,colors)
        key = cv2.waitKey(1)
        if key==27:
            break
    cap.release()


if __name__ == '__main__':
    image = True
    webcam = False
    video = False
    verbose = True

    if webcam:
        if verbose:
            print('starting webcam')
        webcam_detect()


    if image:
        image_detect(img_path)

    if video:
        start_video()





















