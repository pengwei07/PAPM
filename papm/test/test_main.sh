!/bin/bash

SeedList=(2023)
for seed in ${SeedList[@]}
    do
    python main.py --file_link='./datasets/burger_250_101_2_64_64_sorted.h5' --train_bs=16 --val_bs=8 --shuffle=True --model='cno' --loss_path='./figs/1.png' --gpu='cpu' --seed=$seed --test_accumulative_error=True --recording_path='./recording/1.txt' --weight_path='./weights/1.pth' 
    python main.py --file_link='./datasets/burger_250_101_2_64_64_sorted.h5' --train_bs=16 --val_bs=8 --shuffle=True --model='unet' --loss_path='./figs/1.png' --gpu='cpu' --seed=$seed --test_accumulative_error=True --recording_path='./recording/1.txt' --weight_path='./weights/1.pth'
    python main.py --file_link='./datasets/burger_250_101_2_64_64_sorted.h5' --train_bs=16 --val_bs=8 --shuffle=True --model='lstm' --loss_path='./figs/1.png' --gpu='cpu' --seed=$seed --test_accumulative_error=True --recording_path='./recording/1.txt' --weight_path='./weights/1.pth' 
    python main.py --file_link='./datasets/burger_250_101_2_64_64_sorted.h5' --train_bs=16 --val_bs=8 --shuffle=True --model='fno' --loss_path='./figs/1.png' --gpu='cpu' --seed=$seed --test_accumulative_error=True --recording_path='./recording/1.txt' --weight_path='./weights/1.pth'
    python main.py --file_link='./datasets/burger_250_101_2_64_64_sorted.h5' --train_bs=16 --val_bs=8 --shuffle=True --model='res' --loss_path='./figs/1.png' --gpu='cpu' --seed=$seed --test_accumulative_error=True --recording_path='./recording/1.txt' --weight_path='./weights/1.pth' 
    python main.py --file_link='./datasets/burger_250_101_2_64_64_sorted.h5' --train_bs=16 --val_bs=8 --shuffle=True --model='papm' --loss_path='./figs/1.png' --gpu='cpu' --seed=$seed --test_accumulative_error=True --recording_path='./recording/1.txt' --weight_path='./weights/1.pth' 
    python main.py --file_link='./datasets/burger_250_101_2_64_64_sorted.h5' --train_bs=16 --val_bs=8 --shuffle=True --model='percnn' --loss_path='./figs/1.png' --gpu='cpu' --seed=$seed --test_accumulative_error=True --recording_path='./recording/1.txt' --weight_path='./weights/1.pth' 
    python main.py --file_link='./datasets/burger_250_101_2_64_64_sorted.h5' --train_bs=16 --val_bs=8 --shuffle=True --model='ppnn' --loss_path='./figs/1.png' --gpu='cpu' --seed=$seed --test_accumulative_error=True --recording_path='./recording/1.txt' --weight_path='./weights/1.pth' 
    done

