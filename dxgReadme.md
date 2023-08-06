## Requirements

Python 3.8 or later with all ```requirements.txt``` dependencies installed, including `torch>=1.7`. To install run:
```bash
$ pip install -r requirements.txt
```

这里安装的torch, torchvison 会报错，自己安装GPU版本的
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html


mkdir datasets

ln -s /aic/datasets/AIC22_Track1_MTMC_Tracking ./datasets
AIC22_Track1_MTMC_Tracking/ -> /aic/datasets/AIC22_Track1_MTMC_Tracking/

cd detector/yolov5/
wget https://github.com/ultralytics/yolov5/releases/download/v4.0/yolov5x.pt

pip install yacs numba

change sh gen_det.sh -> bash gen_det.sh

yolov5报错,要自己重装cuda版本的torch, torchvision
bash gen_det.sh aic_all.yml
生成了裁切的图片

## codeDetail

全过程`run_all.sh`每一步的输入输出如下：

MCMT_CONFIG_FILE="aic_all.yml"

#### Run Detector.####
cd detector/
python gen_images_aic.py ${MCMT_CONFIG_FILE}
>>>>>>>>将数据集提供的视频转成图片，还使用了数据集提供的ROI遮挡了不用检测的区域，图像命名就是用frame number
detection

cd yolov5/
bash gen_det.sh ${MCMT_CONFIG_FILE}
>>>>>>>>车辆检测， 生成用于特征提取的patch，以及车辆检测的bbox
detect_merge
dets        patch
dets_debug  全图带检测结果
labels      检测结果yolo的txt格式标注

#### Extract reid feautres.####
cd ../../reid/
python extract_image_feat.py "aic_reid1.yml"
python extract_image_feat.py "aic_reid2.yml"
python extract_image_feat.py "aic_reid3.yml"
python merge_reid_feat.py ${MCMT_CONFIG_FILE}
>>>>>>>>车辆特征提取，
detect_reid1
detect_reid2
detect_reid3
三种检测得到的特征进行l2 norm  就是欧式距离
*dets_feat.pkl

#### MOT. ####
cd ../tracker/MOTBaseline
sh run_aic.sh ${MCMT_CONFIG_FILE}
wait
*_mot.txt

#### Get results. ####
cd ../../reid/reid-matching/tools
python trajectory_fusion.py ${MCMT_CONFIG_FILE}
*_mot_feat_break.pkl
python sub_cluster.py ${MCMT_CONFIG_FILE}
python gen_res.py ${MCMT_CONFIG_FILE}





## 跑通之后，再跑出来一个validation/S02的结果







