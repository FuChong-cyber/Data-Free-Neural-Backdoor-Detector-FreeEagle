c_path="D:/datasets/CIFAR10"
g_path="D:/datasets/GTSRB"
i_path="D:/datasets/ImageNetSubset"

# imagenet_subset x resnet50
# cifar10 x vgg16
# gtsrb x google_net

python ../backdoor_attack_simulation/model_training.py --dataset cifar10 --model vgg16 --data_folder ${c_path}
python ../backdoor_attack_simulation/model_training.py --dataset gtsrb --model google_net --data_folder ${g_path}
python ../backdoor_attack_simulation/model_training.py --dataset imagenet_subset --model resnet50 --data_folder ${i_path}
