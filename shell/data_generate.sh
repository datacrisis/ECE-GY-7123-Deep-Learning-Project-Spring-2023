#!/bin/bash

target_name=nissan-titan
target_object=body

root=/home/endeleze/Desktop/Active_NeRF
data_path=$root/data
model_path=$data_path/model/$target_name.blend
output_path=$data_path/images
script=$root/src/capture.py

sphere_samples_num=1000
sphere_offset_ratio=8
camera_location_type=sphere

blender $model_path --background --python $script -- --output_path $output_path --target_name $target_name --target_object $target_object --sphere_samples_num $sphere_samples_num \
--sphere_offset_ratio $sphere_offset_ratio --camera_location_type $camera_location_type --full_sphere


