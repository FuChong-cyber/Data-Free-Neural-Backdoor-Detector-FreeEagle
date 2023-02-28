@echo off
call activate pytorch-gpu

::vgg16 resnet50 google_net
::patched_img blending_img

set c_path=D:/datasets/CIFAR10
set g_path=D:/datasets/GTSRB
set i_path=D:/datasets/ImageNetSubset

::imagenet_subset x resnet50
::cifar10 x vgg16
::gtsrb x google_net

::Agnostic backdoor
::for %%t in (patched_img blending_img) do (
::for /l %%i in (0,1,19) do (
::python ../backdoor_attack_simulation/model_training.py --dataset poisoned_imagenet_subset --model resnet50 --trigger_type %%t --targeted_class %%i --poison_ratio 0.2 --data_folder %i_path%
::)
::)

::Specific backdoor
for %%t in (patched_img blending_img) do (
::0 14
for %%i in (12 18) do (
for /l %%s in (0,1,19) do (
if %%i neq %%s (
python ../backdoor_attack_simulation/model_training.py --dataset poisoned_imagenet_subset --model resnet50 --trigger_type %%t --targeted_class %%i --source_classes %%s --poison_ratio 0.025 --data_folder %i_path%
)))

)
