#!/bin/bash

function terminate_scripts {
    echo "Terminating running scripts..."
    pkill -P $$  
    exit 1
}


trap terminate_scripts SIGINT


# run for our method

echo "part 1"


nohup python -u NP_efficient_main.py -c 0 -seed 0 -m semipde -ss 4 -ts 20 > test1.log 2>&1 &
nohup python -u NP_efficient_main.py -c 1 -seed 0 -m semipde -ss 6 -ts 20 > test2.log 2>&1 &
nohup python -u NP_efficient_main.py -c 6 -seed 0 -m semipde -ss 8 -ts 20 > test3.log 2>&1 &
nohup python -u NP_efficient_main.py -c 3 -seed 0 -m semipde -ss 10 -ts 20 > test4.log 2>&1 &
nohup python -u NP_efficient_main.py -c 4 -seed 0 -m semipde -ss 12 -ts 20 > test5.log 2>&1 &
nohup python -u NP_efficient_main.py -c 5 -seed 0 -m semipde -ss 14 -ts 20 > test6.log 2>&1 &

nohup python -u NP_efficient_main.py -c 7 -seed 0 -m semipde -ss 4 -ts 50 > test7.log 2>&1 &
nohup python -u NP_efficient_main.py -c 8 -seed 0 -m semipde -ss 6 -ts 50 > test8.log 2>&1 &
nohup python -u NP_efficient_main.py -c 9 -seed 0 -m semipde -ss 8 -ts 50 > test9.log 2>&1 &
nohup python -u NP_efficient_main.py -c 10 -seed 0 -m semipde -ss 10 -ts 50 > test10.log 2>&1 &
nohup python -u NP_efficient_main.py -c 11 -seed 0 -m semipde -ss 12 -ts 50 > test11.log 2>&1 &
nohup python -u NP_efficient_main.py -c 12 -seed 0 -m semipde -ss 14 -ts 50 > test12.log 2>&1 &


echo "waiting..."
wait

echo "All scripts have been run successfully."