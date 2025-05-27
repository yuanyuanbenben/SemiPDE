#!/bin/bash

function terminate_scripts {
    echo "Terminating running scripts..."
    pkill -P $$  
    exit 1
}


trap terminate_scripts SIGINT


# run for our method

# echo "part 1"
# nohup python -u NP_main.py -c 6 -n 64 -seed 0 -m semipde > test1.log 2>&1 &
# nohup python -u NP_main.py -c 7 -n 64 -seed 10 -m semipde > test2.log 2>&1 &
# nohup python -u NP_main.py -c 8 -n 64 -seed 20 -m semipde > test3.log 2>&1 &
# nohup python -u NP_main.py -c 9 -n 64 -seed 30 -m semipde > test4.log 2>&1 &
# nohup python -u NP_main.py -c 10 -n 64 -seed 40 -m semipde > test5.log 2>&1 &


echo "part 1"
nohup python -u NP_main.py -c 6 -n 64 -seed 0 > test1.log 2>&1 &
nohup python -u NP_main.py -c 7 -n 64 -seed 10 > test2.log 2>&1 &
nohup python -u NP_main.py -c 8 -n 64 -seed 20 > test3.log 2>&1 &
nohup python -u NP_main.py -c 9 -n 64 -seed 30 > test4.log 2>&1 &
nohup python -u NP_main.py -c 10 -n 64 -seed 40 > test5.log 2>&1 &


echo "waiting..."
wait


echo "All scripts have been run successfully."

