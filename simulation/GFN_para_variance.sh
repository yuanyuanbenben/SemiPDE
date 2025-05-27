#!/bin/bash

function terminate_scripts {
    echo "Terminating running scripts..."
    pkill -P $$  
    exit 1
}


trap terminate_scripts SIGINT


# run for our method
echo "part 1"

nohup python -u GFN_variance.py -c 0 -seed 100 -m semipde -ss 8 -ts 10 -in 2 > var_test.log 2>&1 &
nohup python -u GFN_variance.py -c 1 -seed 150 -m semipde -ss 8 -ts 10 -in 2 > var_test2.log 2>&1 &
nohup python -u GFN_variance.py -c 8 -seed 200 -m semipde -ss 8 -ts 10 -in 2 > var_test4.log 2>&1 &
nohup python -u GFN_variance.py -c 3 -seed 250 -m semipde -ss 8 -ts 10 -in 2 > var_test5.log 2>&1 &
nohup python -u GFN_variance.py -c 4 -seed 300 -m semipde -ss 8 -ts 10 -in 2 > var_test6.log 2>&1 &
nohup python -u GFN_variance.py -c 5 -seed 350 -m semipde -ss 8 -ts 10 -in 2 > var_test7.log 2>&1 &
nohup python -u GFN_variance.py -c 6 -seed 400 -m semipde -ss 8 -ts 10 -in 2 > var_test8.log 2>&1 &
nohup python -u GFN_variance.py -c 7 -seed 450 -m semipde -ss 8 -ts 10 -in 2 > var_test9.log 2>&1 &
nohup python -u GFN_variance.py -c 9 -seed 0 -m semipde -ss 8 -ts 10 -in 2 > var_test10.log 2>&1 &
nohup python -u GFN_variance.py -c 10 -seed 50 -m semipde -ss 8 -ts 10 -in 2 > var_test22.log 2>&1 &

nohup python -u GFN_variance.py -c 0 -seed 125 -m semipde -ss 8 -ts 10 -in 2 > var_test11.log 2>&1 &
nohup python -u GFN_variance.py -c 1 -seed 175 -m semipde -ss 8 -ts 10 -in 2 > var_test12.log 2>&1 &
nohup python -u GFN_variance.py -c 8 -seed 225 -m semipde -ss 8 -ts 10 -in 2 > var_test14.log 2>&1 &
nohup python -u GFN_variance.py -c 3 -seed 275 -m semipde -ss 8 -ts 10 -in 2 > var_test15.log 2>&1 &
nohup python -u GFN_variance.py -c 4 -seed 325 -m semipde -ss 8 -ts 10 -in 2 > var_test16.log 2>&1 &
nohup python -u GFN_variance.py -c 5 -seed 375 -m semipde -ss 8 -ts 10 -in 2 > var_test17.log 2>&1 &
nohup python -u GFN_variance.py -c 6 -seed 425 -m semipde -ss 8 -ts 10 -in 2 > var_test18.log 2>&1 &
nohup python -u GFN_variance.py -c 7 -seed 475 -m semipde -ss 8 -ts 10 -in 2 > var_test19.log 2>&1 &
nohup python -u GFN_variance.py -c 9 -seed 25 -m semipde -ss 8 -ts 10 -in 2 > var_test20.log 2>&1 &
nohup python -u GFN_variance.py -c 10 -seed 75 -m semipde -ss 8 -ts 10 -in 2 > var_test21.log 2>&1 &

# nohup python -u GFN_variance.py -c 9 -seed 0 -m semipde -ss 8 -ts 10 -in 2 > var_test20.log 2>&1 &
# nohup python -u GFN_variance.py -c 10 -seed 50 -m semipde -ss 8 -ts 10 -in 2 > var_test21.log 2>&1 &
# nohup python -u GFN_variance.py -c 11 -seed 0 -m semipde -ss 8 -ts 10 -in 3 > var_test22.log 2>&1 &
# nohup python -u GFN_variance.py -c 12 -seed 50 -m semipde -ss 8 -ts 10 -in 3 > var_test23.log 2>&1 &

# nohup python -u GFN_variance.py -c 9 -seed 25 -m semipde -ss 8 -ts 10 -in 2 > var_test24.log 2>&1 &
# nohup python -u GFN_variance.py -c 10 -seed 75 -m semipde -ss 8 -ts 10 -in 2 > var_test25.log 2>&1 &
# nohup python -u GFN_variance.py -c 11 -seed 25 -m semipde -ss 8 -ts 10 -in 3 > var_test26.log 2>&1 &
# nohup python -u GFN_variance.py -c 12 -seed 75 -m semipde -ss 8 -ts 10 -in 3 > var_test27.log 2>&1 &


echo "waiting..."
wait


echo "All scripts have been run successfully."

