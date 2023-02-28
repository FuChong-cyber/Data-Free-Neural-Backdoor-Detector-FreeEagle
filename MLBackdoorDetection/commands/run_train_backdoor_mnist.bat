@echo off
call activate pytorch-gpu

::vgg16 resnet50 google_net
::patched_img blending_img

set m_path=D:/datasets/

::imagenet_subset x resnet50
::cifar10 x vgg16
::gtsrb x google_net

::Agnostic backdoor
for %%t in (patched_img blending_img filter_img) do (

for /l %%i in (0,1,9) do (
python ../backdoor_attack_simulation/model_training.py --dataset poisoned_mnist --model simple_cnn --trigger_type %%t --targeted_class %%i --poison_ratio 0.2 --data_folder %m_path%
)

)

::Specific backdoor
for %%t in (patched_img blending_img filter_img) do (

for /l %%i in (0,1,9) do (
for /l %%s in (0,1,9) do (
if %%i neq %%s (
python ../backdoor_attack_simulation/model_training.py --dataset poisoned_mnist --model simple_cnn --trigger_type %%t --targeted_class %%i --source_classes %%s --poison_ratio 0.05 --data_folder %m_path%
)))

)