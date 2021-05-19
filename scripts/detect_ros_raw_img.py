#!/usr/bin/env python
# Author: Tejas
# Date: May, 6, 2021
# Purpose: Ros node to detect objects using tensorflow

import os
import sys
import cv2
import numpy as np
from PIL import Image as imgp
import tensorflow as tf

# ROS related imports
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

# Object detection module imports
import object_detection
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# object 3d pose imports
from geometry_msgs.msg import PoseStamped, PointStamped
from sensor_msgs.msg import PointCloud2, CameraInfo
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from image_geometry import PinholeCameraModel
import tf2_ros
import tf2_geometry_msgs


# Disable Tensroflow log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# SET FRACTION OF GPU YOU WANT TO USE HERE
GPU_FRACTION = 0.4

######### Set model here ############
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
# By default models are stored in data/models/
MODEL_PATH = os.path.join(os.path.dirname(
    sys.path[0]), 'data', 'models', MODEL_NAME)
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_PATH + '/frozen_inference_graph.pb'
######### Set the label map file here ###########
LABEL_NAME = 'mscoco_label_map.pbtxt'
# By default label maps are stored in data/labels/
PATH_TO_LABELS = os.path.join(os.path.dirname(
    sys.path[0]), 'data', 'labels', LABEL_NAME)
######### Set the number of classes here #########
NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`,
# we know that this corresponds to `airplane`.  Here we use internal utility functions,
# but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Setting the GPU options to use fraction of gpu that has been set
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = GPU_FRACTION

# Detection


class Detector:

    def __init__(self):

        # Defining the topics
        self.image_topic = "/rgb/image_raw"  # "/camera/rgb/image_color"
        self.depth_topic = "/depth_to_rgb/image_raw"  # "/camera/depth/image"
        self.camera_frame = "zed_link"
        self.camera_topic = "/depth_to_rgb/camera_info"  # "/camera/depth/camera_info"
        self.object_detect = "/object_detected_image"  # "/object_detected_image"
        self.marker_topic = "visualization_marker_array"

        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1, buff_size=2**24)
        self.image_pub = rospy.Publisher(self.object_detect, Image, queue_size=1)
        self.bridge = CvBridge()
        self.sess = tf.compat.v1.Session(graph=detection_graph, config=config)
        self.marker_pub = rospy.Publisher(self.marker_topic, MarkerArray, queue_size=100)

        # Transformations
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(100.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Getting the camera info
        self.camera_info = rospy.wait_for_message(self.camera_topic, CameraInfo)

        # Making Camera Model
        self.cam_model = PinholeCameraModel()
        self.cam_model.fromCameraInfo(self.camera_info)
        rospy.loginfo_once("Ready for object detection")

    def image_cb(self, data):

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = np.asarray(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, num_detections) = self.sess.run([boxes, scores, classes, num_detections],
                                                                     feed_dict={image_tensor: image_np_expanded})

        objects = vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=2)

        if num_detections >= 1:
            # object_count=1

            # Looping over each of detected output
            for i in range(len(objects)):
                # object_count+=1
                image_height, image_width, channels = image_np.shape
                # height_bound = 480
                # width_bound = 640
                # Extracting the bounding box coordinates
                object_id = objects[i][0]
                object_score = objects[i][1]
                object_class = category_index[object_id]
                dimensions = objects[i][2]

                # center coordinates
                cx = int((dimensions[1] + dimensions[3])*image_height/2)
                cy = int((dimensions[0] + dimensions[2])*image_width/2)

                # Converting the center cooridnates to base link frame
                # bounding the center coordinates to image size

                cx = np.clip(cx, 0, image_height)
                cy = np.clip(cy, 0, image_width)

                ######Debug##########
                # depth = rospy.wait_for_message(self.depth_topic, Image)
                # depth_img = imgp.frombytes("F", (depth.width, depth.height), depth.data)
                # lookup = depth_img.load()
                # try:
                #     d = lookup[cx, cy]
                #     l = [cx,cy,d]
                #     print (l)
                # except:
                #     continue

                # print(l)

                # print("centre:"+str(l)+" score:"+str(object_score)+" ID:"+str(object_class))
                # try:
                #
                # except:
                #     rospy.loginfo("***************************failed")
                #     continue
                ######################

                # Find 3d coordinates using bounding box centroid and depth info
                self.distance, self.object = self.uv_to_xyz(cx, cy)

                # cv2.putText(image, "distance = "+str(self.distance),
                #             (dimensions[1] - 10, dimensions[0] - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX,
				#             1,
                #             (255,0,0),  
				#             2)

        # Publish object detected image
        img=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_out=Image()
        try:
            image_out=self.bridge.cv2_to_imgmsg(img, "bgr8")
        except CvBridgeError as e:
            print(e)

        self.image_pub.publish(image_out)

                # Draw markers wrt world transform
                # self.draw_markers(self.object)

    '''
		Function to convert 2D image coordinates to 3D world coordinates using PinHoleCamera Model and Depth Image

		Input -
			1. cx - X coordinate in image frame
			2. cy - Y coordinte in image frame
	'''
    # Function for converting coordinates to base link frame

    def uv_to_xyz(self, cx, cy):

        # Converting to XYZ coordinates
        (x, y, z)=self.cam_model.projectPixelTo3dRay((cx, cy))
        # Normalising
        x, y, z=x/z, y/z, z/z

        # Getting the depth at given coordinates
        depth=rospy.wait_for_message(self.depth_topic, Image)

        depth_img=imgp.frombytes(
            "F", (depth.width, depth.height), depth.data)
        lookup=depth_img.load()

        try:
            d=lookup[cx, cy]
        except:
            d=None

        # Ignoring some depth values
        if d != None:
            # Modifying the coordinates
            x, y, z=x*d, y*d, z*d
            # rospy.loginfo("coordinates: "+str(x)+" , "+str(y)+" , "+str(z))

            # Making Point Stamp Message
            point_wrt_camera=PointStamped()
            point_wrt_camera.header.frame_id=self.camera_frame

            point_wrt_camera.point.x=x
            point_wrt_camera.point.y=y
            point_wrt_camera.point.z=z

            return d, point_wrt_camera

    '''
		Function to visualize markers in Rviz

		Input -
			1. object - point stamped message wrt world coordinate

	'''

    def draw_markers(self, object):

        markerArray=MarkerArray()
        count=0
        MARKERS_MAX=100

        marker=Marker()
        marker.header.frame_id=object.header.frame_id#"world"
        marker.type=marker.SPHERE
        marker.action=marker.ADD
        marker.scale.x=0.2
        marker.scale.y=0.2
        marker.scale.z=0.2
        marker.color.a=1.0
        marker.color.r=0
        marker.color.g=0
        marker.color.b=100
        marker.pose.orientation.w=1.0
        marker.pose.position.x=object.point.x
        marker.pose.position.y=object.point.y
        marker.pose.position.z=object.point.z

        # We add the new marker to the MarkerArray, removing the oldest marker from it when necessary
        if(count > MARKERS_MAX):
            markerArray.markers.pop(0)

        markerArray.markers.append(marker)

        # Renumber the marker IDs
        id=0
        for m in markerArray.markers:
            m.id=id
            id += 1
        # Publish the MarkerArray
        self.marker_pub.publish(markerArray)

        count += 1


def main(args):
    rospy.init_node('detector_node')
    obj=Detector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("ShutDown")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
