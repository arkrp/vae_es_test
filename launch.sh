podman container rm vae_es_test
podman build -t vae_es_test .
podman run -it --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --name vae_es_test vae_es_test $1
