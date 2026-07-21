FROM public.ecr.aws/lambda/microvms:al2023-minimal

RUN dnf install -y \
        amazon-efs-utils \
        bash \
        findutils \
        iproute \
        iptables-nft \
        nfs-utils \
        python3.11 \
        python3.11-pip \
        shadow-utils \
        util-linux \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && groupadd --gid 1000 cayu-agent \
    && useradd --uid 1000 --gid 1000 --no-create-home --shell /sbin/nologin cayu-agent \
    && dnf clean all

ENV CAYU_MICROVM_AGENT_UID=1000 \
    CAYU_MICROVM_AGENT_GID=1000 \
    CAYU_MICROVM_AGENT_NETNS=cayu-agent

WORKDIR /opt/cayu
COPY . /opt/cayu/lambda_microvm_sidecar
RUN python3.11 -m pip install --no-cache-dir \
        -r /opt/cayu/lambda_microvm_sidecar/requirements.txt \
        "botocore>=1.43.44,<2" \
    && install -d -o 1000 -g 1000 /workspace

EXPOSE 8080
CMD ["bash", "/opt/cayu/lambda_microvm_sidecar/entrypoint.sh"]
