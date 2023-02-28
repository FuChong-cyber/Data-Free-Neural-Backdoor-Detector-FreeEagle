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
for %%t in (patched_img blending_img) do (
for /l %%i in (0,1,19) do (
python ../backdoor_attack_simulation/model_training.py --dataset poisoned_imagenet_subset --model resnet50 --trigger_type %%t --targeted_class %%i --poison_ratio 0.2 --data_folder %i_path%
)
for /l %%i in (0,1,9) do (
python ../backdoor_attack_simulation/model_training.py --dataset poisoned_cifar10 --model vgg16 --trigger_type %%t --targeted_class %%i --poison_ratio 0.2 --data_folder %c_path%
)
for /l %%i in (0,1,42) do (
python ../backdoor_attack_simulation/model_training.py --dataset poisoned_gtsrb --model google_net --trigger_type %%t --targeted_class %%i --poison_ratio 0.2 --data_folder %g_path%
)
)

::Specific backdoor
for %%t in (patched_img blending_img) do (

for /l %%i in (0,1,9) do (
for /l %%s in (0,1,9) do (
if %%i neq %%s (
python ../backdoor_attack_simulation/model_training.py --dataset poisoned_cifar10 --model vgg16 --trigger_type %%t --targeted_class %%i --source_classes %%s --poison_ratio 0.05 --data_folder %c_path%
)))

for %%i in (7 8) do (
for /l %%s in (0,1,42) do (
if %%i neq %%s (
python ../backdoor_attack_simulation/model_training.py --dataset poisoned_gtsrb --model google_net --trigger_type %%t --targeted_class %%i --source_classes %%s --poison_ratio 0.01 --data_folder %g_path%
)))

for %%i in (0 14) do (
for /l %%s in (0,1,19) do (
if %%i neq %%s (
python ../backdoor_attack_simulation/model_training.py --dataset poisoned_imagenet_subset --model resnet50 --trigger_type %%t --targeted_class %%i --source_classes %%s --poison_ratio 0.025 --data_folder %i_path%
)))

)