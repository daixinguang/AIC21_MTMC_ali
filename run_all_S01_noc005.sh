MCMT_CONFIG_FILE="aic_all_S01_noc005.yml"
# #### Run Detector.####
# cd detector/
# python gen_images_aic_dxg.py ${MCMT_CONFIG_FILE}

# cd yolov5/
# bash gen_det_dxg.sh ${MCMT_CONFIG_FILE} c001 c002 c003 c004 c005

# #### Extract reid feautres.####
# cd ../../reid/
# python extract_image_feat.py "aic_reid1_S01.yml"
# python extract_image_feat.py "aic_reid2_S01.yml"
# python extract_image_feat.py "aic_reid3_S01.yml"
# python merge_reid_feat_dxg.py ${MCMT_CONFIG_FILE}

# #### MOT. ####
# cd ../tracker/MOTBaseline
# bash run_aic_dxg.sh ${MCMT_CONFIG_FILE} c001 c002 c003 c004 c005
# wait
#### Get results. ####
cd ../../reid/reid-matching/tools
python trajectory_fusion_dxg.py ${MCMT_CONFIG_FILE} # 直接用S01提供的检测跟踪即可。在这个py代码里去掉c005去轨迹融合即可。
python sub_cluster_dxg.py ${MCMT_CONFIG_FILE} # # 直接用S01提供的检测跟踪即可。在这个py代码里去掉c005去轨迹融合即可。
python gen_res.py ${MCMT_CONFIG_FILE}

#### Vis. (optional) ####
# python viz_mot_dxg.py ${MCMT_CONFIG_FILE}
# python viz_mcmt.py ${MCMT_CONFIG_FILE}
