@echo off
call activate pytorch-gpu

set c_path=D:/datasets/CIFAR10
set g_path=D:/datasets/GTSRB
set i_path=D:/datasets/ImageNetSubset

for /l %%i in (0,1,60) do (
python ../backdoor_attack_simulation/model_training.py --dataset imagenet_subset --model resnet50 --data_folder %i_path%
python ../backdoor_attack_simulation/model_training.py --dataset cifar10 --model vgg16 --data_folder %c_path%
python ../backdoor_attack_simulation/model_training.py --dataset gtsrb --model google_net --data_folder %g_path%
)
