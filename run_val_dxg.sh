# 生成validation/S02的检测结果
MCMT_CONFIG_FILE="aic_val_dxg.yml"
#### Run Detector.####
cd detector/ || exit
python gen_images_aic_dxg.py ${MCMT_CONFIG_FILE}

cd yolov5/ || exit
bash gen_det_dxg.sh ${MCMT_CONFIG_FILE}

#### Extract reid feautres.####
cd ../../reid/ || exit
python extract_image_feat.py "aic_reid_dxg1.yml"
python extract_image_feat.py "aic_reid_dxg2.yml"
python extract_image_feat.py "aic_reid_dxg3.yml"
python merge_reid_feat_dxg.py ${MCMT_CONFIG_FILE}

#### MOT. ####
cd ../tracker/MOTBaseline || exit
bash run_aic_dxg.sh ${MCMT_CONFIG_FILE}
wait
#### Get results. ####
cd ../../reid/reid-matching/tools || exit
python trajectory_fusion_dxg.py ${MCMT_CONFIG_FILE}
python sub_cluster_dxg.py ${MCMT_CONFIG_FILE}
python gen_res.py ${MCMT_CONFIG_FILE}

#### Vis. (optional) ####
# python viz_mot.py ${MCMT_CONFIG_FILE}
# python viz_mcmt.py ${MCMT_CONFIG_FILE}
