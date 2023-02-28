c_path="D:/datasets/CIFAR10"
g_path="D:/datasets/GTSRB"
i_path="D:/datasets/ImageNetSubset"

# imagenet_subset x resnet50
# cifar10 x vgg16
# gtsrb x google_net

triggers=(patched_img blending_img)

# Agnostic backdoor
for trigger in "${triggers[@]}"; do

  for targeted_class in $(seq 0 19); do
    python ../backdoor_attack_simulation/model_training.py --dataset poisoned_imagenet_subset --model resnet50 --trigger_type "${trigger}" --targeted_class "${targeted_class}" --poison_ratio 0.2 --data_folder ${i_path}
  done

  for targeted_class in $(seq 0 9); do
    python ../backdoor_attack_simulation/model_training.py --dataset poisoned_cifar10 --model vgg16 --trigger_type "${trigger}" --targeted_class "${targeted_class}" --poison_ratio 0.2 --data_folder ${c_path}
  done

  for targeted_class in $(seq 0 42); do
    python ../backdoor_attack_simulation/model_training.py --dataset poisoned_gtsrb --model google_net --trigger_type "${trigger}" --targeted_class "${targeted_class}" --poison_ratio 0.2 --data_folder ${g_path}
  done

done

# Specific backdoor
for trigger in "${triggers[@]}"; do

  for targeted_class in $(seq 0 9); do
    for source_class in $(seq 0 9); do
      if [ "${targeted_class}" != "${source_class}" ]; then
        python ../backdoor_attack_simulation/model_training.py --dataset poisoned_cifar10 --model vgg16 --trigger_type "${trigger}" --targeted_class "${targeted_class}" --source_classes "${source_class}" --poison_ratio 0.05 --data_folder ${c_path}
      fi
    done
  done

  for targeted_class in 7 8; do
    for source_class in $(seq 0 42); do
      if [ "${targeted_class}" != "${source_class}" ]; then
        python ../backdoor_attack_simulation/model_training.py --dataset poisoned_gtsrb --model google_net --trigger_type "${trigger}" --targeted_class "${targeted_class}" --source_classes "${source_class}" --poison_ratio 0.01 --data_folder ${g_path}
      fi
    done
  done

  for targeted_class in 0 14 12 18; do
    for source_class in $(seq 0 19); do
      if [ "${targeted_class}" != "${source_class}" ]; then
        python ../backdoor_attack_simulation/model_training.py --dataset poisoned_imagenet_subset --model resnet50 --trigger_type "${trigger}" --targeted_class "${targeted_class}" --source_classes "${source_class}" --poison_ratio 0.025 --data_folder ${i_path}
      fi
    done
  done

done
