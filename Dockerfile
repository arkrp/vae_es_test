FROM docker.io/rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.7.1
WORKDIR /app
COPY /app /app
CMD ["/opt/venv/bin/python", "/app/main.py"]

