#!/bin/bash

function terminate_scripts {
    echo "Terminating running scripts..."
    pkill -P $$  
    exit 1
}


trap terminate_scripts SIGINT


# run for our method

echo "part 1"


# nohup python -u RD_efficient_main.py -c 1 -seed 0 -m semipde -ss 8 -ts 5 > test13.log 2>&1 &
# nohup python -u RD_efficient_main.py -c 2 -seed 0 -m semipde -ss 12 -ts 5 > test14.log 2>&1 &
# nohup python -u RD_efficient_main.py -c 3 -seed 0 -m semipde -ss 16 -ts 5 > test15.log 2>&1 &
# nohup python -u RD_efficient_main.py -c 4 -seed 0 -m semipde -ss 24 -ts 5 > test16.log 2>&1 &
# nohup python -u RD_efficient_main.py -c 5 -seed 0 -m semipde -ss 32 -ts 5 > test17.log 2>&1 &
# nohup python -u RD_efficient_main.py -c 6 -seed 0 -m semipde -ss 40 -ts 5 > test18.log 2>&1 &

# nohup python -u RD_efficient_main.py -c 7 -seed 0 -m semipde -ss 8 -ts 10 > test7.log 2>&1 &
# nohup python -u RD_efficient_main.py -c 8 -seed 0 -m semipde -ss 12 -ts 10 > test8.log 2>&1 &
# nohup python -u RD_efficient_main.py -c 9 -seed 0 -m semipde -ss 16 -ts 10 > test9.log 2>&1 &
# nohup python -u RD_efficient_main.py -c 10 -seed 0 -m semipde -ss 24 -ts 10 > test10.log 2>&1 &
# nohup python -u RD_efficient_main.py -c 11 -seed 0 -m semipde -ss 32 -ts 10 > test11.log 2>&1 &
# nohup python -u RD_efficient_main.py -c 12 -seed 0 -m semipde -ss 40 -ts 10 > test12.log 2>&1 &

# nohup python -u RD_efficient_main.py -c 10 -seed 0 -m semipde -ss 8 -ts 20 > test11.log 2>&1 &
# nohup python -u RD_efficient_main.py -c 11 -seed 0 -m semipde -ss 12 -ts 20 > test12.log 2>&1 &
# nohup python -u RD_efficient_main.py -c 12 -seed 0 -m semipde -ss 16 -ts 20 > test13.log 2>&1 &
# nohup python -u RD_efficient_main.py -c 13 -seed 0 -m semipde -ss 24 -ts 20 > test14.log 2>&1 &
# nohup python -u RD_efficient_main.py -c 14 -seed 0 -m semipde -ss 32 -ts 20 > test15.log 2>&1 &
# nohup python -u RD_efficient_main.py -c 15 -seed 0 -m semipde -ss 40 -ts 20 > test16.log 2>&1 &

# nohup python -u RD_efficient_main.py -c 10 -seed 40 -m semipde -ss 8 -ts 50 > test19.log 2>&1 &
# nohup python -u RD_efficient_main.py -c 11 -seed 40 -m semipde -ss 12 -ts 50 > test20.log 2>&1 &
# nohup python -u RD_efficient_main.py -c 12 -seed 40 -m semipde -ss 16 -ts 50 > test21.log 2>&1 &
# nohup python -u RD_efficient_main.py -c 13 -seed 40 -m semipde -ss 24 -ts 50 > test22.log 2>&1 &
# nohup python -u RD_efficient_main.py -c 14 -seed 40 -m semipde -ss 32 -ts 50 > test23.log 2>&1 &
# nohup python -u RD_efficient_main.py -c 15 -seed 40 -m semipde -ss 40 -ts 50 > test24.log 2>&1 &

echo "waiting..."
wait

echo "All scripts have been run successfully."