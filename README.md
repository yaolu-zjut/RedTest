# RedTest: Towards Measuring Redundancy in Deep Neural Networks Effectively

## 1.Model training
The following command will train a vgg11 on CIFAR10.
```bash
python train.py --gpu 1 --arch cvgg11 --set cifar10 --lr 0.01 --batch_size 256 --weight_decay 0.005 --epochs 150 --lr_decay_step 50,100  --num_classes 10
```

## 2.MSRS Measurement
The following command will calculate the MSRS of ResNet20 on CIFAR10.
```bash
python calculating_MSRS.py --gpu 0 --arch resnet20 --set cifar10 --num_classes 10 --batch_size 256 --pretrained  --evaluate
```

## 3.Redundancy-aware NAS
The following command will calculate the MSRS of all candidate architectures in NATS-Bench on CIFAR10.
```bash
python NAS_MSRS.py --gpu 2 --set cifar10 --batch_size 256 --num_classes 10
```

## 4.redundancy-aware layer pruning 
There are pruned models in get_new_model.py. The following command will return the similarities between two adjacent layers
```bash
python CKA_guide_laye_pruning.py --gpu 0 --arch resnet56 --set cifar10 --num_classes 10 --batch_size 256 --pretrained  --evaluate
```
