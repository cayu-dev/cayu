FROM public.ecr.aws/lambda/microvms:al2023-minimal

RUN dnf install -y \
        amazon-efs-utils \
        bash \
        findutils \
        nfs-utils \
        python3.11 \
        python3.11-pip \
        util-linux \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && dnf clean all

WORKDIR /opt/cayu
COPY requirements.txt /opt/cayu/requirements.txt
RUN python3.11 -m pip install --no-cache-dir \
        -r /opt/cayu/requirements.txt \
        "botocore>=1.43.44,<2"

RUN mkdir -p /opt/cayu/lambda_microvm_sidecar /workspace
COPY __init__.py app.py supervisor.py \
     /opt/cayu/lambda_microvm_sidecar/
COPY entrypoint.sh /opt/cayu/entrypoint.sh

EXPOSE 8080
CMD ["bash", "/opt/cayu/entrypoint.sh"]
