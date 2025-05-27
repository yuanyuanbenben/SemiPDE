#!/bin/bash

function terminate_scripts {
    echo "Terminating running scripts..."
    pkill -P $$  
    exit 1
}


trap terminate_scripts SIGINT


# run for our method

echo "part 1"




nohup python -u GFN_efficient_main.py -c 0 -seed 160 -m semipde -ss 8 -ts 10 > test1.log 2>&1 &
nohup python -u GFN_efficient_main.py -c 1 -seed 260 -m semipde -ss 8 -ts 10 > test2.log 2>&1 &
nohup python -u GFN_efficient_main.py -c 2 -seed 360 -m semipde -ss 8 -ts 10 > test3.log 2>&1 &
nohup python -u GFN_efficient_main.py -c 3 -seed 460 -m semipde -ss 8 -ts 10 > test4.log 2>&1 &

nohup python -u GFN_efficient_main.py -c 4 -seed 180 -m semipde -ss 8 -ts 10 > test5.log 2>&1 &
nohup python -u GFN_efficient_main.py -c 5 -seed 280 -m semipde -ss 8 -ts 10 > test6.log 2>&1 &
nohup python -u GFN_efficient_main.py -c 6 -seed 380 -m semipde -ss 8 -ts 10 > test7.log 2>&1 &
nohup python -u GFN_efficient_main.py -c 7 -seed 480 -m semipde -ss 8 -ts 10 > test8.log 2>&1 &
# nohup python -u GFN_efficient_main.py -c 4 -seed 90 -m semipde -ss 8 -ts 10 > test17.log 2>&1 & 

# nohup python -u GFN_efficient_main.py -c 4 -seed 20 -m semipde -ss 3 -ts 20 > test1.log 2>&1 &
# nohup python -u GFN_efficient_main.py -c 5 -seed 20 -m semipde -ss 4 -ts 20 > test2.log 2>&1 &
# nohup python -u GFN_efficient_main.py -c 6 -seed 20 -m semipde -ss 5 -ts 20 > test3.log 2>&1 &
# nohup python -u GFN_efficient_main.py -c 7 -seed 20 -m semipde -ss 6 -ts 20 > test4.log 2>&1 &
# nohup python -u GFN_efficient_main.py -c 8 -seed 20 -m semipde -ss 7 -ts 20 > test5.log 2>&1 &
# nohup python -u GFN_efficient_main.py -c 9 -seed 20 -m semipde -ss 8 -ts 20 > test6.log 2>&1 &

# nohup python -u GFN_efficient_main.py -c 10 -seed 45 -m semipde -ss 3 -ts 20 > test7.log 2>&1 &
# nohup python -u GFN_efficient_main.py -c 11 -seed 45 -m semipde -ss 4 -ts 20 > test8.log 2>&1 &
# nohup python -u GFN_efficient_main.py -c 12 -seed 45 -m semipde -ss 5 -ts 20 > test9.log 2>&1 &
# nohup python -u GFN_efficient_main.py -c 13 -seed 45 -m semipde -ss 6 -ts 20 > test10.log 2>&1 &
# nohup python -u GFN_efficient_main.py -c 14 -seed 45 -m semipde -ss 7 -ts 20 > test11.log 2>&1 &
# nohup python -u GFN_efficient_main.py -c 15 -seed 45 -m semipde -ss 8 -ts 20 > test12.log 2>&1 &

echo "waiting..."
wait

echo "All scripts have been run successfully."