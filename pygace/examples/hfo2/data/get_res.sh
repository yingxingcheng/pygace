#!/bin/bash

cal_tag=100003
output_dir=ce_ball
if [ ! -d $output_dir ];then
    mkdir $output_dir
fi

for x in `ls -d [0-9]*`;do 
    sub_dir=${cal_tag}${x}
    if [ ! -d ${output_dir}/${sub_dir} ];then
        mkdir ${output_dir}/${sub_dir}
    fi

    cd $x 
    extract_energy
    cd ../ 
    cp ${x}/energy ${x}/str.out ${output_dir}/${sub_dir} 
done
