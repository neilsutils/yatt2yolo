out="output"

rm ${out}/*.txt
rm ${out}/*.png

output_file="results"

python3 detect-yatt.py --weights /home/thesourorange/yolov5/runs/exp3/weights/best.pt --device cpu --conf 0.4 --out output test-set.zip

echo "recno,weights_file,image_file,conf_thres,iou_thres,tag_cls_name,tag_x,tag_y,tag_w,tag_h,conf,found,correct,radius,distance,bbox_cls_name,bbox_x, bbox_y, bbox_w, bbox_h" > ${output_file}.csv

cat ${out}/*.txt >> ${output_file}.csv

