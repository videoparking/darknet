import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))

import darknet as dn
#import pdb
import json

dn.set_gpu(0)
net = dn.load_net(b"cfg/yolov3.cfg", b"yolov3.weights", 0)
meta = dn.load_meta(b"cfg/coco.data")

while True:
    line = sys.stdin.readline().strip()
    r = dn.detect(net, meta, line.encode("utf-8"))
    print("%s: %s", line, json.dumps(r))
