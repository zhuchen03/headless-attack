function runexp {
subs_chk=${1}
subs_net=${2}
target_chk=${3}
target_net=${4}
cname=${5}
seed=${6}
gpu=${7}
nvars=${8}
gaussstd=${9}
pgd_steps=${10}
pgd_lr=${11}
eps=${12}

    echo ${eps}
    python attack.py --subs-chk-name ${subs_chk} --subs-net ${subs_net} --target-net ${target_net} --target-chk-name ${target_chk} --centroid-out-name ${cname}  --seed ${seed} --gpu ${gpu} --overwrite --n-variants ${nvars} --gauss-std ${gaussstd} --pgd-steps ${pgd_steps} --pgd-lr ${pgd_lr} --eps ${eps} #> avi-logs/subs${subs_net}-${subs_chk}-tgt${target_net}-${target_chk}-seed${seed}-nvars${nvars}-gstd${gaussstd}-pgdsteps${pgd_steps}-pgdlr${pgd_lr}-4800samples.log 2>&1 
    python attack.py --pgd --subs-chk-name ${subs_chk} --subs-net ${subs_net} --target-net ${target_net} --target-chk-name ${target_chk} --centroid-out-name ${cname}  --seed ${seed} --gpu ${gpu} --overwrite --n-variants ${nvars} --gauss-std ${gaussstd} --pgd-steps ${pgd_steps} --pgd-lr ${pgd_lr} --eps ${eps} #> avi-logs/pgd-subs${subs_net}-${subs_chk}-tgt${target_net}-${target_chk}-seed${seed}-nvars${nvars}-gstd${gaussstd}-pgdsteps${pgd_steps}-pgdlr${pgd_lr}-4800samples.log 2>&1 

}

# runexp       subs_chk     subs_net      target_chk                                        target_net  cname               seed   gpu  nvars   gaussstd    steps    lr     eps
#runexp     ckpt-%s-4800.t7  ResNet18      ckpt-%s-4800-dp0.000-droplayer0.000-seed1226.t7   ResNet18    centroids/init.pth  987123 0   4        0.0         20      0.05    1
#runexp     ckpt-%s-4800.t7  ResNet18      ckpt-%s-4800-dp0.000-droplayer0.000-seed1226.t7   ResNet18    centroids/init.pth  000000 0   4        0.0         20      0.05    1
#runexp     ckpt-%s-4800.t7  ResNet18      ckpt-%s-4800-dp0.000-droplayer0.000-seed1226.t7   ResNet18    centroids/init.pth  300000 0   4        0.0         20      0.05    1
#runexp     ckpt-%s-4800.t7  ResNet18      ckpt-%s-4800-dp0.000-droplayer0.000-seed1226.t7   ResNet18    centroids/init.pth  900003 0   4        0.0         20      0.05    1
#runexp     ckpt-%s-4800.t7  ResNet18      ckpt-%s-4800-dp0.000-droplayer0.000-seed1226.t7   ResNet18    centroids/init.pth  982213 0   4        0.0         20      0.05    1

#runexp     ckpt-%s-4800.t7  ResNet18      ckpt-%s-4800-dp0.000-droplayer0.000-seed1226.t7   ResNet18    centroids/init.pth  987123 1   4        0.0         20      0.05    2
#runexp     ckpt-%s-4800.t7  ResNet18      ckpt-%s-4800-dp0.000-droplayer0.000-seed1226.t7   ResNet18    centroids/init.pth  000000 1   4        0.0         20      0.05    2
#runexp     ckpt-%s-4800.t7  ResNet18      ckpt-%s-4800-dp0.000-droplayer0.000-seed1226.t7   ResNet18    centroids/init.pth  300000 1   4        0.0         20      0.05    2
#runexp     ckpt-%s-4800.t7  ResNet18      ckpt-%s-4800-dp0.000-droplayer0.000-seed1226.t7   ResNet18    centroids/init.pth  900003 1   4        0.0         20      0.05    2
#runexp     ckpt-%s-4800.t7  ResNet18      ckpt-%s-4800-dp0.000-droplayer0.000-seed1226.t7   ResNet18    centroids/init.pth  982213 1   4        0.0         20      0.05    2
#
runexp     ckpt-%s-4800.t7  ResNet18      ckpt-%s-4800-dp0.000-droplayer0.000-seed1226.t7   ResNet18    centroids/init.pth  987123 2   4        0.0         20      0.05    4
runexp     ckpt-%s-4800.t7  ResNet18      ckpt-%s-4800-dp0.000-droplayer0.000-seed1226.t7   ResNet18    centroids/init.pth  000000 2   4        0.0         20      0.05    4
runexp     ckpt-%s-4800.t7  ResNet18      ckpt-%s-4800-dp0.000-droplayer0.000-seed1226.t7   ResNet18    centroids/init.pth  300000 2   4        0.0         20      0.05    4
runexp     ckpt-%s-4800.t7  ResNet18      ckpt-%s-4800-dp0.000-droplayer0.000-seed1226.t7   ResNet18    centroids/init.pth  900003 2   4        0.0         20      0.05    4
runexp     ckpt-%s-4800.t7  ResNet18      ckpt-%s-4800-dp0.000-droplayer0.000-seed1226.t7   ResNet18    centroids/init.pth  982213 2   4        0.0         20      0.05    4

runexp     ckpt-%s-4800.t7  ResNet18      ckpt-%s-4800-dp0.000-droplayer0.000-seed1226.t7   ResNet18    centroids/init.pth  987123 2   4        0.0         20      0.05    8
runexp     ckpt-%s-4800.t7  ResNet18      ckpt-%s-4800-dp0.000-droplayer0.000-seed1226.t7   ResNet18    centroids/init.pth  000000 2   4        0.0         20      0.05    8
runexp     ckpt-%s-4800.t7  ResNet18      ckpt-%s-4800-dp0.000-droplayer0.000-seed1226.t7   ResNet18    centroids/init.pth  300000 2   4        0.0         20      0.05    8
runexp     ckpt-%s-4800.t7  ResNet18      ckpt-%s-4800-dp0.000-droplayer0.000-seed1226.t7   ResNet18    centroids/init.pth  900003 2   4        0.0         20      0.05    8
runexp     ckpt-%s-4800.t7  ResNet18      ckpt-%s-4800-dp0.000-droplayer0.000-seed1226.t7   ResNet18    centroids/init.pth  982213 2   4        0.0         20      0.05    8

