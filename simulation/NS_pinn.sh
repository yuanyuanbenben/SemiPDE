#!/bin/bash

function terminate_scripts {
    echo "Terminating running scripts..."
    pkill -P $$  
    exit 1
}


trap terminate_scripts SIGINT


for i in {1..10}
do
CUDA_VISIBLE_DEVICES=11 nohup python -u NS_PINN_main.py -n 64 -seed $((i*5-5)) -l 0.01 -s 0.1 > pinn_test1.log 2>&1 &
CUDA_VISIBLE_DEVICES=12 nohup python -u NS_PINN_main.py -n 64 -seed $((i*5-4)) -l 0.01 -s 0.1 > pinn_test2.log 2>&1 &
CUDA_VISIBLE_DEVICES=13 nohup python -u NS_PINN_main.py -n 64 -seed $((i*5-3)) -l 0.01 -s 0.1 > pinn_test3.log 2>&1 &
CUDA_VISIBLE_DEVICES=14 nohup python -u NS_PINN_main.py -n 64 -seed $((i*5-2)) -l 0.01 -s 0.1 > pinn_test4.log 2>&1 &
CUDA_VISIBLE_DEVICES=15 nohup python -u NS_PINN_main.py -n 64 -seed $((i*5-1)) -l 0.01 -s 0.1 > pinn_test5.log 2>&1 &
wait
done


echo "All scripts have been run successfully."

