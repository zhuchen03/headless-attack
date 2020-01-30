function runexp {
gpu=${1}
topk=${2}
steps=${3}
lr=${4}
eps=${5}
seed=${6}
chk=${7}

nohup \
    python attack-headless.py --batchsize 64 --target-chk-name ${chk} --gpu ${gpu} --label-k ${topk} --eps ${eps} --seed ${seed} --pgd-lr ${lr} --pgd-steps ${steps} --maximize  \
    > logs/headless-resnet50-k${topk}-steps${steps}-lr${lr}-eps${eps}-seed${seed}-maximize.log   2>&1 &
    
}

# runexp    gpu     topk    steps   lr      eps     seed        chk
runexp      0       1       20     0.05    8       1234        chks/resnet50-lr0.4-last.pth
runexp      1       1       20     0.05    8       8023       chks/resnet50-lr0.4-last.pth
runexp      2       1       20     0.05    8       9017       chks/resnet50-lr0.4-last.pth
runexp      3       1       20     0.05    8       2333        chks/resnet50-lr0.4-last.pth

