#!/bin/bash

function terminate_scripts {
    echo "Terminating running scripts..."
    pkill -P $$  
    exit 1
}


trap terminate_scripts SIGINT


# run for our method

# echo "part 1"
# nohup python -u NS_main.py -c 6 -seed 0 -m semipde -n 64 > test11.log 2>&1 &
# nohup python -u NS_main.py -c 7 -seed 10 -m semipde -n 64 > test12.log 2>&1 &
# nohup python -u NS_main.py -c 8 -seed 20 -m semipde -n 64 > test13.log 2>&1 &
# nohup python -u NS_main.py -c 9 -seed 30 -m semipde -n 64 > test14.log 2>&1 &
# nohup python -u NS_main.py -c 10 -seed 40 -m semipde -n 64 > test15.log 2>&1 &

# echo "part 1"
# nohup python -u NS_main.py -c 11 -seed 0 -m semipde -n 64 -s 0.5 > test16.log 2>&1 &
# nohup python -u NS_main.py -c 12 -seed 10 -m semipde -n 64 -s 0.5 > test17.log 2>&1 &
# nohup python -u NS_main.py -c 13 -seed 20 -m semipde -n 64 -s 0.5 > test18.log 2>&1 &
# nohup python -u NS_main.py -c 14 -seed 30 -m semipde -n 64 -s 0.5 > test19.log 2>&1 &
# nohup python -u NS_main.py -c 15 -seed 40 -m semipde -n 64 -s 0.5 > test20.log 2>&1 &

echo "part 1"
nohup python -u NS_main.py -c 11 -seed 0 -n 64 > test16.log 2>&1 &
nohup python -u NS_main.py -c 12 -seed 10 -n 64 > test17.log 2>&1 &
nohup python -u NS_main.py -c 13 -seed 20 -n 64 > test18.log 2>&1 &
nohup python -u NS_main.py -c 14 -seed 30 -n 64 > test19.log 2>&1 &
nohup python -u NS_main.py -c 15 -seed 40 -n 64 > test20.log 2>&1 &



echo "waiting..."
wait


echo "All scripts have been run successfully."

