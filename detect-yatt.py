import argparse

import torch.backends.cudnn as cudnn

from models.experimental import *
from utils.datasets import *
from utils.utils import *

import zipfile
import io
import json
import os

RADIUS = float(.010)

def replace_all(text, subs, replacement):
    for sub in subs:
        text = text.replace(sub, replacement)
    return text

def process_tags(zf, name):
    tags = []

    with io.TextIOWrapper(zf.open(name), encoding="utf-8") as f:
        data = f.read()

        annotations = json.loads(data)

        if 'dimensions' not in annotations:
            return tags

        if 'regions' not in annotations:
            return tags

        dimensions = annotations["dimensions"]

        if 'width' not in dimensions:
            return tags

        if 'height' not in dimensions:
            return tags

        width = float(dimensions["width"])
        height = float(dimensions["height"])

        for region in annotations['regions']:
            if 'region_attributes' not in region:
                continue

            if 'shape_attributes' not in region:
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

            tag = [tag_name, bbox_x_centre, bbox_y_centre, bbox_width, bbox_height]

            print("    Tag: '%s' %.8f %.8f %.8f %.8f" % (tag_name, bbox_x_centre, bbox_y_centre, bbox_width, bbox_height))

            tags.append(tag)

    return tags

def reshape_image(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

    # loads 1 image from dataset, returns img, original hw, resized hw

def load_image(path, img_size, augment):
    img = cv2.imread(path)  # BGR
    assert img is not None, 'Image Not Found ' + path
    h0, w0 = img.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size

    if r != 1:  # always resize down, only resize up if training with augmentation
       interp = cv2.INTER_AREA if r < 1 and not augment else cv2.INTER_LINEAR
       img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)

    return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized

def transform_image(img0, img_size):
    # Padded resize
    img = reshape_image(img0, new_shape=img_size)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    return img, img0 

def check_for_hit(p1, p2, t1, t2, radius):
    distance = math.sqrt((p1 - t1)**2 + (p2 - t2)**2)
    return distance <= radius, distance 

def test_tag(test_tags, resolved_tags, cls, p1, p2):
    index = 0

    for tag in test_tags:

        resolved = (len(list(filter (lambda x : x == index, resolved_tags))) > 0)

        if resolved:
            next

        found, distance = check_for_hit(p1, p2, float(tag[1]), float(tag[2]), RADIUS)

        print(("Testing: [%d] : %s = %s : "  + ("%f " * 6) + " %r %r") % (index, cls, tag[0], p1, p2, float(tag[1]), float(tag[2]), RADIUS, distance, found, resolved)) 

        if found and not resolved: 
           resolved_tags.append(index)

           return RADIUS, distance, found, (cls==tag[0]), tag[0], tag[1], tag[2], tag[3], tag[4] 
        else:
           index += 1

    return RADIUS, -1, False, False, '-1', '-1', '-1', '-1', '-1' 
       
def check_tags(results_file, test_tags, resolved_tags, model_name, tail, names, conf_thres, iou_thres):
    index = 0

    for tag in test_tags:
        resolved = (len(list(filter (lambda x : x == index, resolved_tags))) > 0)
    
        if not resolved:
           results_file.write(('1,%s,%s,%f,%f,%s,' + ('%d,' * 5) + '%f,' + '%r,%r,%f,%s,%s,' + ('%s,' * 3) +  '\n') %
                              (model_name, tail, conf_thres, iou_thres, '', -1, -1, -1, -1, 0, False, False, 
                               RADIUS, 0.0, tag[0], tag[1], tag[2], tag[3], tag[4]))

        index += 1

def detect(image_file_name, image_tags, save_img=False):
    out, weights, view_img, save_txt, imgsz, augment, work,  = \
    opt.output, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.augment, opt.work

    # Initialize
    device = torch_utils.select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
 
    source = os.path.join(work, image_file_name)

    print(source, image_tags)

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

    image, (h0, w0), (h, w) = load_image(source, imgsz, augment) 

    _, model_name = os.path.split(weights[0]) 

    print("Source: '%s', (%d, %d), (%d, %d)" % (source, h0, w0, h, w))
    head, tail = os.path.split(source)

    base = os.path.splitext(tail)[0]

    print("Input: head = '%s', tail = '%s', base = %s" % (head, tail, base))

    results_file  = open(os.path.join(out, base + ".txt"), "w")

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    print(names)

    colors = [[random.randint(0, 254) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    img, img0 = transform_image(image, imgsz)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
       img = img.unsqueeze(0)

    # Inference
    t1 = torch_utils.time_synchronized()
    pred = model(img, augment=augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    t2 = torch_utils.time_synchronized()
    resolved_tags = [] 

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        s, im0 = '', img0

        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

        if det is not None and len(det):
           # Rescale boxes from img_size to im0 size
           det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

           # Print results
           for c in det[:, -1].unique():
               n = (det[:, -1] == c).sum()  # detections per class
               s += '%g %ss, ' % (n, names[int(c)])  # add to string

           # Write results
           for *xyxy, conf, cls in det:

                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                label = '%s %.2f' % (names[int(cls)], conf)

                radius, distance, found, correct, bbox_label, bbox_x, bbox_y, bbox_w, bbox_h = test_tag(image_tags, resolved_tags, names[int(cls)], xywh[0], xywh[1])

                print(('%g,' * 5 + '%f') % (cls, *xywh, conf))  # label format
                
                results_file.write(('0,%s,%s,%f,%f,%s,' + ('%g,' * 4) + '%f,' + '%r,%r,%f,%f,%s,' + ('%s,' * 3) + '%s\n') % 
                                   (model_name, tail, opt.conf_thres, opt.iou_thres, names[int(cls)], *xywh, 
                                    conf, found, correct, radius, distance,
                                    bbox_label, bbox_x, bbox_y, bbox_w, bbox_h))

                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

        # Print time (inference + NMS)
        print('%sDone. (%.3fs)' % (s, t2 - t1))

    check_tags(results_file, image_tags, resolved_tags, model_name, tail, names, opt.conf_thres, opt.iou_thres)

    cv2.imwrite(os.path.join(out, tail), img0)

    results_file.close()

    print('Done. (%.3fs)' % (time.time() - t0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  
    parser.add_argument('--work', type=str, default='/tmp', help='Temp Folder')  
    parser.add_argument('--test', type=str, default='inference/test', help='test set')  
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('archives', nargs='+', type=str, help='yatt file')
    opt = parser.parse_args()
    print(opt)

    tags = {} 
    images = {} 

    for archive in opt.archives:
        with zipfile.ZipFile(archive, 'r') as zf:
             for name in zf.namelist():
                id = replace_all(name, ['.json', '.png'], '')
    
                if name.endswith('.json'): 
                   print("Including tags: '%s'" % name)
                   tags[id] = process_tags(zf, name)
                elif name.endswith('.png'): 
                   zf.extract(name, path=opt.work)
                   images[id] = name

    for tag in tags:
        with torch.no_grad():
             detect(images[tag], tags[tag])
