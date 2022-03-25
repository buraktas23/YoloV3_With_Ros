#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import rospy
import numpy as np
import cv2
import sys
import main
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import os



bridge = CvBridge()

#########load yolo #####
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

######## load_img ############
def load_image(img_path):

    img = cv2.imread(img_path)    
    img = cv2.resize(img, None, fx=0.3, fy=0.25)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)

    height,width,channels = img.shape

    return height,width,channels,img

##### detect object ########
def detect_objec(net,img,output_layers):

    blob = cv2.dnn.blobFromImage(img,scalefactor=0.00392, size=(320,320),mean=(0,0,0),swapRB=True,crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    return blob,outputs

###### get box dimantion ##########
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

###### draw label ############
def draw_labels(boxes,class_ids,confs,classes,img,colors):

    indexes = cv2.dnn.NMSBoxes(boxes,confs,0.5,0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            print('color: ', color)
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


#### take frame from by using ros #####
def take_frame(data):
	print('image is taken')
	


	global bridge

	cv_frame = bridge.imgmsg_to_cv2(data, "bgr8") 

	frame = np.array(cv_frame, dtype=np.uint8)

	classes,output_layers,colors,model = load_yolo()
	height,width,channels = frame.shape
	blob,outputs = detect_objec(model,frame,output_layers)
	boxes,class_ids,confs = get_box_dimention(outputs,height,width)
	draw_labels(boxes,class_ids,confs,classes,frame,colors)
   

	#cv2.imshow('camera', frame)
	

	if cv2.waitKey(1) & 0xFF == ord('q'):
		rospy.signal_shutdown('closing...')

		
		


def main(args):

	rospy.init_node('camera', anonymous = True)
	rospy.Subscriber('/usb_cam/image_raw', Image,take_frame)


	try:
	  rospy.spin()

	except KeyboardInterrupt:

	  print('closing..')
	  cv2.destroyAllWindows()

if __name__ == '__main__':
	
	
	main(sys.argv)
	
