# Deep Quadruplet Fingerprinting (DQF) Attack
:warning: experimental - PLEASE BE CAREFUL. Intended for reasearch purposes only.

The source code and dataset are used to demonstrate the Deep Quadruplet Fingerprinting (DQF) model.

# Platform
- python: 3.8
- Pytorch: 1.8

## architecture

DQF/
├── datasets   
	├── section5.2.1  
	├── ... 
├── savedmodels
	├── AWF775.pth
├── dataset_opw.py
├── dataset_qua_semihard.py
└── dataset.py
├── lib.py
├── loss.py
└── model.py
├── modules.py
├── test_opw.py            # For open_world evaluations
└── test.py                # For closed_world evaluations
└── train_qua_semihard.py  # For pre-training


##  Example: WF attack on similar but mutually exclusive datasets

```
python /test.py --ckp model_path --testdata_path data_path --num_way SOCP --num_shot num_training --num_query num_test

```
* model_path: the path of the trained model
* data_path: the path of the classification dataset
* SOCP: the size of the classification problem
* num_training: the number of training sampels per website in a support set
* num_test: the number of test samples per website in a query set 

.