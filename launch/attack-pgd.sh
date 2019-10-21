function runexp {
subs_chk=${1}
subs_net=${2}
target_chk=${3}
target_net=${4}
cname=${5}

    python attack.py --subs-chk-name ${subs_chk} --subs-net ${subs_net} --target-net ${target_net} --target-chk-name ${target_chk} --centroid-out-name ${cname} --pgd

}

# runexp       subs_chk     subs_net      target_chk                                        target_net  cname
runexp     ckpt-%s-4800.t7  ResNet18      ckpt-%s-4800-dp0.000-droplayer0.000-seed1226.t7   ResNet18    centroids/init.pth
