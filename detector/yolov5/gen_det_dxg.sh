# seqs=(c001 c002 c003 c004 c005) # S01
# seqs=(c006 c007 c008 c009) # S02
# seqs=(c041 c042 c043 c044 c045 c046) # S06


gpu_id=0
for seq in ${@:2}
do
    CUDA_VISIBLE_DEVICES=${gpu_id} python detect2img.py --name ${seq} --weights yolov5x.pt --conf 0.1 --agnostic --save-txt --save-conf --img-size 1280 --classes 2 5 7 --cfg_file $1&
    gpu_id=$(($gpu_id+1))
done
wait
