ADDENDUM
PYTHON SCRIPT FOR TRAINING THE OBJECT DETECTION MODEL
import torch

from IPython.display import Image, clear_output  # to display images
from utils.downloads import attempt_download  # to download models/datasets

# clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))


from roboflow import Roboflow
rf = Roboflow(api_key="jd4Ij8Ld5yN5MQ00kP6N")
project = rf.workspace("finalyearproject-axe8j").project("masked-armed-bandits")
dataset = project.version(8).download("yolov5")

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/yolov5
#after following the link above, recieve python code with these fields filled in
#from roboflow import Roboflow
#rf = Roboflow(api_key="YOUR API KEY HERE")
#project = rf.workspace().project("YOUR PROJECT")
#dataset = project.version("YOUR VERSION").download("yolov5")

# Commented out IPython magic to ensure Python compatibility.
# this is the YAML file Roboflow wrote for us that we are loading into this notebook with our data
# %cat {dataset.location}/data.yaml

"""# Define Model Configuration and Architecture

We will write a yaml script that defines the parameters for our model like the number of classes, anchors, and each layer.

You do not need to edit these cells, but you may.
"""

# define number of classes based on YAML
import yaml
with open(dataset.location + "/data.yaml", 'r') as stream:
   num_classes = str(yaml.safe_load(stream)['nc'])

# Commented out IPython magic to ensure Python compatibility.
#this is the model configuration we will use for our tutorial
# %cat /content/yolov5/models/yolov5s.yaml

#customize iPython writefile so we can write variables
from IPython.core.magic import register_line_cell_magic

@register_line_cell_magic
def writetemplate(line, cell):
   with open(line, 'w') as f:
       f.write(cell.format(**globals()))

# Commented out IPython magic to ensure Python compatibility.
# %%writetemplate /content/yolov5/models/custom_yolov5s.yaml
#
# # YOLOv5 🚀 by Ultralytics, GPL-3.0 license
#
# # Parameters
# nc: 3  # number of classes
# depth_multiple: 0.33  # model depth multiple
# width_multiple: 0.50  # layer channel multiple
# anchors:
#   - [10,13, 16,30, 33,23]  # P3/8
#   - [30,61, 62,45, 59,119]  # P4/16
#   - [116,90, 156,198, 373,326]  # P5/32
#
# # YOLOv5 v6.0 backbone
# backbone:
#   # [from, number, module, args]
#   [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
#    [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
#    [-1, 3, C3, [128]],
#    [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
#    [-1, 6, C3, [256]],
#    [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
#    [-1, 9, C3, [512]],
#    [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
#    [-1, 3, C3, [1024]],
#    [-1, 1, SPPF, [1024, 5]],  # 9
#   ]
#
# # YOLOv5 v6.0 head
# head:
#   [[-1, 1, Conv, [512, 1, 1]],
#    [-1, 1, nn.Upsample, [None, 2, 'nearest']],
#    [[-1, 6], 1, Concat, [1]],  # cat backbone P4
#    [-1, 3, C3, [512, False]],  # 13
#
#    [-1, 1, Conv, [256, 1, 1]],
#    [-1, 1, nn.Upsample, [None, 2, 'nearest']],
#    [[-1, 4], 1, Concat, [1]],  # cat backbone P3
#    [-1, 3, C3, [256, False]],  # 17 (P3/8-small)
#
#    [-1, 1, Conv, [256, 3, 2]],
#    [[-1, 14], 1, Concat, [1]],  # cat head P4
#    [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)
#
#    [-1, 1, Conv, [512, 3, 2]],
#    [[-1, 10], 1, Concat, [1]],  # cat head P5
#    [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)
#
#    [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
#   ]

"""# Train Custom YOLOv5 Detector

### Next, we'll fire off training!


Here, we are able to pass a number of arguments:
- **img:** define input image size
- **batch:** determine batch size
- **epochs:** define the number of training epochs. (Note: often, 3000+ are common here!)
- **data:** set the path to our yaml file
- **cfg:** specify our model configuration
- **weights:** specify a custom path to weights. (Note: you can download weights from the Ultralytics Google Drive [folder](https://drive.google.com/open?id=1Drs_Aiu7xx6S-ix95f9kNsA6ueKRpN2J))
- **name:** result names
- **nosave:** only save the final checkpoint
- **cache:** cache images for faster training
"""

# Commented out IPython magic to ensure Python compatibility.
# # train yolov5s on custom data for 100 epochs
# # time its performance
# %%time
# %cd /content/yolov5/
# !python train.py --img 416 --batch 16 --epochs 100 --data {dataset.location}/data.yaml --cfg ./models/custom_yolov5s.yaml --weights '' --name yolov5s_results  --cache
 
