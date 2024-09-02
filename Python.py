from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput

net = detectNet(
    model="ssd-mobilenet.onnx",
    labels="labels.txt",
    input_blob="input_0",
    output_cvg="scores",
    output_bbox="boxes",
    threshold=0.5
)
camera = videoSource("/dev/video0")
display = videoOutput("webrtc://@:8554/output")

while True:
    img = camera.Capture()

    if img is None: # capture timeout
        continue

    detections = net.Detect(img)
    for d in detections:
      #print(d)
      label=net.GetClassDesc(d.ClassID)
      if label=="Deer": 
        print ("There is a deer")
      if label=="Dog":
        print ("there is a dog")  
    
    display.Render(img)