#!/bin/bash

function terminate_scripts {
    echo "Terminating running scripts..."
    pkill -P $$  
    exit 1
}


trap terminate_scripts SIGINT


# run for our method

echo "part 1"
nohup python -u RD_main.py -c 7 -n 64 -seed 0 -s 0.1 -mod no_func -m semipde > test17.log 2>&1 &
nohup python -u RD_main.py -c 8 -n 64 -seed 10 -s 0.1 -mod no_func -m semipde > test18.log 2>&1 &
nohup python -u RD_main.py -c 9 -n 64 -seed 20 -s 0.1 -mod no_func -m semipde > test19.log 2>&1 &
nohup python -u RD_main.py -c 10 -n 64 -seed 30 -s 0.1 -mod no_func -m semipde > test20.log 2>&1 &
nohup python -u RD_main.py -c 11 -n 64 -seed 40 -s 0.1 -mod no_func -m semipde > test21.log 2>&1 &

# nohup python -u RD_main.py -c 7 -n 64 -seed 0 -s 0.5 -mod no_func > test22.log 2>&1 &
# nohup python -u RD_main.py -c 8 -n 64 -seed 10 -s 0.5 -mod no_func > test23.log 2>&1 &
# nohup python -u RD_main.py -c 9 -n 64 -seed 20 -s 0.5 -mod no_func > test24.log 2>&1 &
# nohup python -u RD_main.py -c 10 -n 64 -seed 30 -s 0.5 -mod no_func > test25.log 2>&1 &
# nohup python -u RD_main.py -c 11 -n 64 -seed 40 -s 0.5 -mod no_func > test26.log 2>&1 &


echo "waiting..."
wait


echo "All scripts have been run successfully."

