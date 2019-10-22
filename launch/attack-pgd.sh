function runexp {
subs_chk=${1}
subs_net=${2}
target_chk=${3}
target_net=${4}
cname=${5}
seed=${6}
gpu=${7}

nohup \
    python attack.py --subs-chk-name ${subs_chk} --subs-net ${subs_net} --target-net ${target_net} --target-chk-name ${target_chk} --centroid-out-name ${cname} --pgd --seed ${seed} --gpu ${7} \
    > logs/pgd-subs${subs_net}-${subs_chk}-tgt${target_net}-${target_chk}-seed${seed}.log 2>&1 &

}

# runexp       subs_chk     subs_net      target_chk                                        target_net  cname                seed  gpu
runexp     ckpt-%s-4800.t7  ResNet18      %s_cifar10_adv.t7   ResNet18    centroids/init.pth  987123 0
#runexp     ckpt-%s-4800.t7  ResNet18      ckpt-%s-4800-dp0.000-droplayer0.000-seed1226.t7   ResNet18    centroids/init.pth  897565 1
#runexp     ckpt-%s-4800.t7  ResNet18      ckpt-%s-4800-dp0.000-droplayer0.000-seed1226.t7   ResNet18    centroids/init.pth  546532 2
#runexp     ckpt-%s-4800.t7  ResNet18      ckpt-%s-4800-dp0.000-droplayer0.000-seed1226.t7   ResNet18    centroids/init.pth  243580 3

