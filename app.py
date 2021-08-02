from Package import social_distancing_configuration as config
from Package.object_detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import imutils
import cv2
import os
from flask import Flask, render_template, Response , request , redirect , url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip("\n")
# derive the paths to YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

# load our yolo object detector trained on COCO dataset(80 classes)
# COCO 80 classes
print("[INFO] Loading YOLO from disk..")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

if config.USE_GPU:
    print("[INFO] setting preferable backend and target to CUDA..")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
# determine only the output layer names that we need from yolo

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")

global writer
writer = None

vio = 2

@app.route('/',methods=['GET', 'POST'])
def index():
    return render_template('new.html')


app.config["UPLOAD_FOLDER"]= "static/vid/"
@app.route('/action_page',methods=['POST','GET'])
def action_page():
    if request.method=="POST":
        if request.files:
            video = request.files["video"]
            filename = secure_filename(video.filename)
            video.save(os.path.join(app.config["UPLOAD_FOLDER"],video.filename))
            print("Video Saved")
            print(filename)
            return render_template('index.html',filename=filename)
    return redirect(request.url)

@app.route('/webcam/0',methods=['POST','GET'])
def webcam():
    print("inside")
    filename = 0
    return render_template('index.html',filename=filename)

"""@app.route('/display/<filename>')
def display(filename):
    print("running")
    print('vid/'+filename)
    return redirect(url_for('static', filename='vid/' + filename),code=301) """

#vs = cv2.VideoCapture(0)
def gen(name):
    if name=="0":
        vs = cv2.VideoCapture(0)
    else:
        vs = cv2.VideoCapture(name)
    print(name + "1")
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()
        # if the frame was not grabbed , then we have reached the end
        # of the stream
        if not grabbed:
            print("notfound")
            break
        # resize the frame and then detect people (and only people) in it
        frame = imutils.resize(frame, width=700)
        results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))
        # initialize the set of indexes that violate the minimum social distance
        violate = set()
        if len(results) >= 2:
            # extract all centroids from the results and compute the
            # euclidean distance between all pairs of the centroids
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")

            # loop over the upper triangular of the distance matrix
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    if D[i, j] < config.MIN_DISTANCE:
                        # update our violation set with the indexes of
                        # the centroid pairs
                        violate.add(i)
                        violate.add(j)
        
        # loop over the results
        for (i, (prob, bbox, centroid)) in enumerate(results):
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)  # green for everything
            if i in violate:
                color = (0, 0, 255)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 1)
            # draw the total number of social distancing violations on the
            # output frame

        vio = len(violate)
        alrt = "Location is Safe"
        if len(results)!= 0: 
            alrt = "Location is {:.2f} % unsafe".format((len(violate)/len(results))*100)
        text = "Total:{}".format(len(results))
        text2 = "Safe:{}".format(len(results)-len(violate))
        text3 = "Unsafe:{}".format(len(violate))
        
        
        if len(violate)>(len(results)//2):
            cv2.putText(frame, alrt, (180,150), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)


        cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 3)
        cv2.putText(frame, text2, (240, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 3)
        cv2.putText(frame, text3, (450, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
        cv2.imwrite("1.jpg", frame)
        k = cv2.waitKey(0)
        if k & 0xFF == ord('q'):
            break
        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

        #k = cv2.waitKey(0) & 0xFF
        #print(k)
        #if k == 255:  # close on ESC key
        #    cv2.destroyAllWindows()
    vs.release()

@app.route('/video_feed/<filename>',methods=['GET', 'POST'])
def video_feed(filename):
    return Response(gen(filename),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
    #vs.release()
    cv2.destroyAllWindows()