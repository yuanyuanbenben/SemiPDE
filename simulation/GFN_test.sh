#!/bin/bash

function terminate_scripts {
    echo "Terminating running scripts..."
    pkill -P $$  
    exit 1
}


trap terminate_scripts SIGINT


# run for our method

# echo "part 1"
# nohup python -u GFN_main.py -c 1 -n 64 -seed 0 -m semipde > test1.log 2>&1 &
# nohup python -u GFN_main.py -c 2 -n 64 -seed 10 -m semipde > test2.log 2>&1 &
# nohup python -u GFN_main.py -c 3 -n 64 -seed 20 -m semipde > test3.log 2>&1 &
# nohup python -u GFN_main.py -c 4 -n 64 -seed 30 -m semipde > test4.log 2>&1 &
# nohup python -u GFN_main.py -c 5 -n 64 -seed 40 -m semipde > test5.log 2>&1 &
# echo "waiting..."
# wait

echo "part 1"
nohup python -u GFN_main.py -c 7 -n 64 -seed 5 -s 0.1 -mod no_func -m semipde > test17.log 2>&1 &
nohup python -u GFN_main.py -c 8 -n 64 -seed 15 -s 0.1 -mod no_func -m semipde > test18.log 2>&1 &
nohup python -u GFN_main.py -c 9 -n 64 -seed 25 -s 0.1 -mod no_func -m semipde > test19.log 2>&1 &
nohup python -u GFN_main.py -c 10 -n 64 -seed 35 -s 0.1 -mod no_func -m semipde > test20.log 2>&1 &
nohup python -u GFN_main.py -c 11 -n 64 -seed 45 -s 0.1 -mod no_func -m semipde > test21.log 2>&1 &

nohup python -u GFN_main.py -c 12 -n 64 -seed 5 -s 0.5 -mod no_func -m semipde > test22.log 2>&1 &
nohup python -u GFN_main.py -c 13 -n 64 -seed 15 -s 0.5 -mod no_func -m semipde > test23.log 2>&1 &
nohup python -u GFN_main.py -c 14 -n 64 -seed 25 -s 0.5 -mod no_func -m semipde > test24.log 2>&1 &
nohup python -u GFN_main.py -c 15 -n 64 -seed 35 -s 0.5 -mod no_func -m semipde > test25.log 2>&1 &
nohup python -u GFN_main.py -c 2 -n 64 -seed 45 -s 0.5 -mod no_func -m semipde > test26.log 2>&1 &
echo "waiting..."
wait

echo "All scripts have been run successfully."

