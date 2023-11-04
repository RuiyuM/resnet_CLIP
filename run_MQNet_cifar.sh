#! /bin/bash

#methods=("active_query" "test_query")

methods=("MQNet")
#methods=("BADGE_sampling")

for method in ${methods[@]};
do
	for j in 600 800
	do
		for i in 10
		do
#			CUDA_VISIBLE_DEVICES=0 python MQNet.py --gpu 0 --k $i --save-dir log_8_15/ --weight-cent 0 --query-strategy $method --init-percent 8 --known-class 40 --query-batch $j --seed 1 --model ResNet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset Tiny-Imagenet &
#			CUDA_VISIBLE_DEVICES=1 python MQNet.py --gpu 1 --k $i --save-dir log_8_15/ --weight-cent 0 --query-strategy $method --init-percent 8 --known-class 40 --query-batch $j --seed 2 --model ResNet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset Tiny-Imagenet &
#			CUDA_VISIBLE_DEVICES=0 python MQNet.py --gpu 3 --k $i --save-dir log_8_15/ --weight-cent 0 --query-strategy $method --init-percent 8 --known-class 40 --query-batch $j --seed 3 --model ResNet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset Tiny-Imagenet &

			CUDA_VISIBLE_DEVICES=1 python MQNet.py --gpu 1 --k $i --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 8 --known-class 20 --query-batch $j --seed 1 --model ResNet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar100 &
			CUDA_VISIBLE_DEVICES=0 python MQNet.py --gpu 0 --k $i --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 8 --known-class 20 --query-batch $j --seed 2 --model ResNet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar100 &
			CUDA_VISIBLE_DEVICES=3 python MQNet.py --gpu 3 --k $i --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 8 --known-class 20 --query-batch $j --seed 3 --model ResNet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar100 &

			CUDA_VISIBLE_DEVICES=0 python MQNet.py --gpu 0 --k $i --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 1 --known-class 2 --query-batch $j --seed 1 --model ResNet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar10 &
			CUDA_VISIBLE_DEVICES=1 python MQNet.py --gpu 1 --k $i --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 1 --known-class 2 --query-batch $j --seed 2 --model ResNet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar10 &
			CUDA_VISIBLE_DEVICES=3 python MQNet.py --gpu 3 --k $i --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 1 --known-class 2 --query-batch $j --seed 3 --model ResNet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar10 &
			wait
		done
	done
done




#CUDA_VISIBLE_DEVICES=0 python MQNet.py --gpu 0 --k 10 --save-dir log_AL/ --weight-cent 0 --query-strategy MQNet --init-percent 8 --known-class 40 --query-batch 400 --seed 1 --model ResNet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset Tiny-Imagenet




#CUDA_VISIBLE_DEVICES=8 python AL_center_temperature.py --gpu 8 --k 10 --save-dir log_AL/ --weight-cent 0 --query-strategy $method --init-percent 1 --known-class 2 --query-batch $j --seed 1 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar10 




#CUDA_VISIBLE_DEVICES=2 python AL_center_temperature.py --gpu 2 --k 10 --save-dir log_AL/ --weight-cent 0 --query-strategy uncertainty --init-percent 1 --known-class 2 --query-batch 400 --seed 1 --model resnet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar10 


#CUDA_VISIBLE_DEVICES=2 python MQNet.py --gpu 2 --k 10 --save-dir log_AL/ --weight-cent 0 --query-strategy MQNet --init-percent 1 --known-class 2 --query-batch 400 --seed 1 --model ResNet18 --known-T 0.5 --unknown-T 0.5 --modelB-T 1 --dataset cifar10 
