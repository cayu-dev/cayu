FROM public.ecr.aws/lambda/microvms:al2023-minimal

RUN dnf install -y python3.11 python3.11-pip bash \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && dnf clean all

WORKDIR /opt/cayu
COPY requirements.txt /opt/cayu/requirements.txt
RUN python3.11 -m pip install --no-cache-dir -r /opt/cayu/requirements.txt

RUN mkdir -p /opt/cayu/lambda_microvm_sidecar
COPY __init__.py app.py supervisor.py /opt/cayu/lambda_microvm_sidecar/
COPY entrypoint.sh /opt/cayu/entrypoint.sh
RUN mkdir -p /workspace

EXPOSE 8080
CMD ["bash", "/opt/cayu/entrypoint.sh"]
