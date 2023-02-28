# Data-Free-Neural-Backdoor-Detector-FreeEagle
A data-free backdoor detector for deep neural networks.

## Quick Start
### Train trojaned and benign models.

Use 'MLBackdoorDetection/backdoor_attack_simulation/train_all_models.py' to train benign and trojaned models, including models trojaned with the agnostic/specific backdoor, the patch/blending/filter/natural trigger.

To train models trojaned with the composite backdoor, you can use the implementation for composite backdoor at: https://github.com/TemporaryAcc0unt/composite-attack

### Inspect the trained trojaned and benign models.
Use our backdoor detector to inpsect the above trained models, i.e., compute one anomaly metric value and generate one inspection result image for each model. Here are some examples of inspection result image of trojaned models, with the abnormal class pairs highlighted in yellow.

The inspection results will be stored to .csv files.

![image](https://github.com/FuChong-cyber/label-inference-attacks/blob/main/illustration_keep_predict_loss.png)

### Analyze the inspection results.
Analyze the inspection results via 'MLBackdoorDetection/analyze_result_csv.py'.
Manual configuration is needed, e.g., setting the variable 'original_dataset_name' to 'gtsrb' if you want to check the defense performance on the GTSRB dataset.

