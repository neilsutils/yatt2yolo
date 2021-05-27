import argparse
import zipfile
import io
import json
import os
import errno
import yaml
import random

def conv_yaml(nc):
 return f"""#parameters
nc: {nc}  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple

# anchors
anchors:
  - [116,90, 156,198, 373,326]  # P5/32
  - [30,61, 62,45, 59,119]  # P4/16
  - [10,13, 16,30, 33,23]  # P3/8

# YOLOv5 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Focus, [64, 3]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, BottleneckCSP, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 9, BottleneckCSP, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, BottleneckCSP, [512]],
   [-1, 1, Conv, [1024, 3, 2]], # 7-P5/32
   [-1, 1, SPP, [1024, [5, 9, 13]]],
  ]

# YOLOv5 head
head:
  [[-1, 3, BottleneckCSP, [1024, False]],  # 9

   [-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, BottleneckCSP, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, BottleneckCSP, [256, False]],
   [-1, 1, nn.Conv2d, [na * (nc + 5), 1, 1]],  # 18 (P3/8-small)

   [-2, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, BottleneckCSP, [512, False]],
   [-1, 1, nn.Conv2d, [na * (nc + 5), 1, 1]],  # 22 (P4/16-medium)

   [-2, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, BottleneckCSP, [1024, False]],
   [-1, 1, nn.Conv2d, [na * (nc + 5), 1, 1]],  # 26 (P5/32-large)

   [[], 1, Detect, [nc, anchors]],  # Detect(P5, P4, P3)
  ]
"""

def process_tags(opt, zf, name, classes, output):
    with io.TextIOWrapper(zf.open(name), encoding="utf-8") as f:
        data = f.read()

        annotations = json.loads(data)

        if 'dimensions' not in annotations:
            print("Error: '%s' dimensions not found" % (name))
            return

        if 'regions' not in annotations:
            print("Error: '%s' regions not found" % (name))
            return

        dimensions = annotations["dimensions"]

        if 'width' not in dimensions:
            print("Error: '%s' dimensions does not contain - width" % (name))
            return

        if 'height' not in dimensions:
            print("Error: '%s' dimensions does not contain - height" % (name))
            return

        file_base_name = os.path.splitext(annotations["filename"])[0]

        coco_tag_file = open(os.path.join(output, "%s%s" % (file_base_name, '.txt')), "w")

        print ("Processing - '%s', Len: %d" % (name, len(data)))

        width = float(dimensions["width"])
        height = float(dimensions["height"])

        for region in annotations['regions']:
            if 'region_attributes' not in region:
                print("Error: '%s' regions does not contain - region_attributes" % (name)) 
                continue

            if 'shape_attributes' not in region:
                print("Error: '%s' regions does not contain - shape_attributes" % (name)) 
                continue

            region_attributes = region['region_attributes']  
            shape_attributes = region['shape_attributes']  

            shape = shape_attributes['name'] 

            if 'tag' not in region_attributes:
                continue

            if shape != 'rect':
                continue
            
            tag_name = region_attributes['tag']

            shape_x = shape_attributes['x']
            shape_y = shape_attributes['y']
            shape_width = shape_attributes['width']
            shape_height = shape_attributes['height']

            bbox_x_centre = float(float(shape_x + shape_width/2)/width)
            bbox_y_centre = float(float(shape_y + shape_height/2)/height)
            bbox_width = float(shape_width/width)
            bbox_height = float(shape_height/height)

            if tag_name not in classes:
                classes.append(tag_name)

            class_id = classes.index(tag_name)

            coco_tag_file.write("%d %.8f %.8f %.8f %.8f\n" % (class_id, bbox_x_centre, bbox_y_centre, bbox_width, bbox_height))

            print("Tag [%s]: %d %.8f %.8f %.8f %.8f\n" % (name, class_id, bbox_x_centre, bbox_y_centre, bbox_width, bbox_height))

        coco_tag_file.close()
       
def create_directory(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            return
        else:
            raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str, nargs='+', help='yatt archives')  # Input Archive 
    parser.add_argument('--path', type=str, default='train', help='training dataset path')  # Dataset Path 
    parser.add_argument('--yaml', type=str, default='yolo', help='yaml prefix')  # YAML File
    parser.add_argument('--split', type=int, default=80, help='Training/Validation split')  # Training Validation Split 

    opt = parser.parse_args()
    print(opt)

    classes = []
    files = {}

    p = int(opt.split)/100

    train_images_path = os.path.join(opt.path, "data", "images", "train")
    val_images_path = os.path.join(opt.path, "data", "images", "valid")

    train_labels_path = os.path.join(opt.path, "data", "labels", "train")
    val_labels_path = os.path.join(opt.path, "data", "labels", "valid")

    create_directory(train_images_path)
    create_directory(val_images_path)

    create_directory(train_labels_path)
    create_directory(val_labels_path)

    for source in opt.source: 
        with zipfile.ZipFile(source, 'r') as zf:
             for name in zf.namelist():
                 file_base_name = os.path.splitext(name)[0]

                 if file_base_name not in files:
                      if random.random() <=p :
                          files[file_base_name] = { 
                                  'images' : train_images_path,
                                  'labels' : train_labels_path
                                  } 
                      else:
                          files[file_base_name] = { 
                                  'images' : val_images_path,
                                  'labels' : val_labels_path
                                  } 

                 if name.endswith('.json'):
                    process_tags(opt, zf, name, classes, files[file_base_name]['labels'])
                 else:
                    zf.extract(name, files[file_base_name]['images']) 

    
# Generating Yaml    
    print("Info: '%s' writing YAML file" % (opt.yaml))
    yaml_structure = {
                       'train' : train_images_path, 
                       'val' : val_images_path, 
                       'nc' : len(classes),
                       'names' : classes
                    }

    with open("%s.yaml" % opt.yaml, 'w') as yaml_file:
         yaml.dump(yaml_structure, yaml_file, default_flow_style=False) 

    with open("%s_conv.yaml" % opt.yaml, "w") as conv_file:
         conv_file.write(conv_yaml(len(classes)))

         


    
