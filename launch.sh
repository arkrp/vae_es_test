docker container rm vae_es_test
docker build -t vae_es_test .
docker run -it --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --name vae_es_test vae_es_test $1
