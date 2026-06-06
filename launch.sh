podman container rm vae_es_test
podman build -t vae_es_test .
podman run -it\
    --device /dev/kfd:/dev/kfd \
    --device /dev/dri:/dev/dri \
    --group-add keep-groups \
    --runtime crun \
    --name vae_es_test \
    vae_es_test $1
#--cap-add=SYS_PTRACE \
#--security-opt apparmor=unconfined \
#--device /dev/kfd:/dev/kfd \
#--device /dev/dri/renderD128:/dev/dri/renderD128 \
#--device /dev/dri/card1:/dev/dri/card1 \
#-e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
#--security-opt unmask=/sys/dev \
