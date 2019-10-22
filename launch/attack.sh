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

    nohup \
    python attack.py --subs-chk-name ${subs_chk} --subs-net ${subs_net} --target-net ${target_net} --target-chk-name ${target_chk} --centroid-out-name ${cname}  --seed ${seed} --gpu ${gpu} --overwrite --n-variants ${nvars} --gauss-std ${gaussstd} --pgd-steps ${pgd_steps} --pgd-lr ${pgd_lr} --eps ${eps} \
    > logs/subs${subs_net}-${subs_chk}-tgt${target_net}-${target_chk}-seed${seed}-nvars${nvars}-gstd${gaussstd}-pgdsteps${pgd_steps}-pgdlr${pgd_lr}-4800samples.log 2>&1 &

}

# runexp       subs_chk     subs_net      target_chk                                        target_net  cname               seed   gpu  nvars   gaussstd    steps    lr     eps
runexp     ckpt-%s-4800.t7  ResNet18      ckpt-%s-4800-dp0.000-droplayer0.000-seed1226.t7   ResNet18    centroids/init.pth  987123 0   4        0.0         40      0.05    1
#runexp     ckpt-%s-4800.t7  ResNet18      ckpt-%s-4800-dp0.000-droplayer0.000-seed1226.t7   ResNet18    centroids/init.pth  987123 1   4        0.0         60      0.05
#runexp     ckpt-%s-4800.t7  ResNet18      ckpt-%s-4800-dp0.000-droplayer0.000-seed1226.t7   ResNet18    centroids/init.pth  987123 2   4        0.0         80      0.05
#runexp     ckpt-%s-4800.t7  ResNet18      ckpt-%s-4800-dp0.000-droplayer0.000-seed1226.t7   ResNet18    centroids/init.pth  987123 3   4        0.0         100      0.05
