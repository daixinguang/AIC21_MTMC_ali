#!/usr/bin/env bash
echo run $0
echo cfg $1


gpu_id=0
for seq in ${@:2:5}
do
    echo param $seq
    echo $(($gpu_id+1))
done

# for seq in ${@:5}
# do
#     echo param2 $seq
# done