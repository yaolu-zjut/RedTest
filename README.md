# RedTest: Towards Measuring Redundancy in Deep Neural Networks Effectively

## Model training
The following command will train a vgg11 on CIFAR10.
```bash
python train.py --gpu 1 --arch cvgg11 --set cifar10 --lr 0.01 --batch_size 256 --weight_decay 0.005 --epochs 150 --lr_decay_step 50,100  --num_classes 10
```
