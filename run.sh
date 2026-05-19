CUDA_VISIBLE_DEVICES=1 python train.py configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py --cfg-options corrupt="cloudy"

python test.py \
  configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga.py \
  work_dirs/unbiased_teacher_oriented_rcnn_selftraining_cga/latest.pth \
  --eval mAP \
  --show-dir vis_cloudy \
  --show-score-thr 0.3 \
  --cfg-options corrupt="cloudy"

python test.py configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftaining_cga_rsar.py \
  work_dirs/unbiased_teacher_oriented_rcnn_selftaining_cga_rsar/latest.pth --eval mAP \
   --show-dir vis_rsar_smoketest

