#!/bin/bash

function terminate_scripts {
    echo "Terminating running scripts..."
    pkill -P $$  
    exit 1
}


trap terminate_scripts SIGINT


for i in {1..10}
do
CUDA_VISIBLE_DEVICES=11 nohup python -u GFN_PINN_main.py -n 64 -seed $((i*5-5)) --lr 0.005 -l 0.5 -s 0.5 -mod no_func > pinn_test1.log 2>&1 &
CUDA_VISIBLE_DEVICES=12 nohup python -u GFN_PINN_main.py -n 64 -seed $((i*5-4)) --lr 0.005 -l 0.5 -s 0.5 -mod no_func > pinn_test2.log 2>&1 &
CUDA_VISIBLE_DEVICES=13 nohup python -u GFN_PINN_main.py -n 64 -seed $((i*5-3)) --lr 0.005 -l 0.5 -s 0.5 -mod no_func > pinn_test3.log 2>&1 &
CUDA_VISIBLE_DEVICES=14 nohup python -u GFN_PINN_main.py -n 64 -seed $((i*5-2)) --lr 0.005 -l 0.5 -s 0.5 -mod no_func > pinn_test4.log 2>&1 &
CUDA_VISIBLE_DEVICES=15 nohup python -u GFN_PINN_main.py -n 64 -seed $((i*5-1)) --lr 0.005 -l 0.5 -s 0.5 -mod no_func > pinn_test5.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u GFN_PINN_main.py -n 64 -seed $((i*5-5)) --lr 0.005 -l 0.1 -s 0.1 -mod no_func > pinn_test6.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python -u GFN_PINN_main.py -n 64 -seed $((i*5-4)) --lr 0.005 -l 0.1 -s 0.1 -mod no_func > pinn_test7.log 2>&1 &
CUDA_VISIBLE_DEVICES=8 nohup python -u GFN_PINN_main.py -n 64 -seed $((i*5-3)) --lr 0.005 -l 0.1 -s 0.1 -mod no_func > pinn_test8.log 2>&1 &
CUDA_VISIBLE_DEVICES=9 nohup python -u GFN_PINN_main.py -n 64 -seed $((i*5-2)) --lr 0.005 -l 0.1 -s 0.1 -mod no_func > pinn_test9.log 2>&1 &
CUDA_VISIBLE_DEVICES=10 nohup python -u GFN_PINN_main.py -n 64 -seed $((i*5-1)) --lr 0.005 -l 0.1 -s 0.1 -mod no_func > pinn_test10.log 2>&1 &
wait
done


echo "All scripts have been run successfully."

