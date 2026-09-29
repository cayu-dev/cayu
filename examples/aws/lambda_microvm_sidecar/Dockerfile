FROM public.ecr.aws/lambda/microvms:al2023-minimal

RUN dnf install -y \
        amazon-efs-utils \
        bash \
        findutils \
        git \
        gzip \
        iproute \
        iptables-nft \
        nfs-utils \
        python3.11 \
        python3.11-pip \
        shadow-utils \
        tar \
        util-linux \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && groupadd --gid 1000 cayu-agent \
    && useradd --uid 1000 --gid 1000 --no-create-home --shell /sbin/nologin cayu-agent \
    && dnf clean all

# Pinned ripgrep for coding search tools; the digest is checked before install.
ARG RIPGREP_VERSION=14.1.1
RUN case "$(uname -m)" in \
        aarch64) asset="ripgrep-${RIPGREP_VERSION}-aarch64-unknown-linux-gnu" \
            && digest="c827481c4ff4ea10c9dc7a4022c8de5db34a5737cb74484d62eb94a95841ab2f" ;; \
        x86_64) asset="ripgrep-${RIPGREP_VERSION}-x86_64-unknown-linux-musl" \
            && digest="4cf9f2741e6c465ffdb7c26f38056a59e2a2544b51f7cc128ef28337eeae4d8e" ;; \
        *) echo "unsupported architecture $(uname -m)" >&2 && exit 1 ;; \
    esac \
    && curl -fsSL -o /tmp/ripgrep.tar.gz \
        "https://github.com/BurntSushi/ripgrep/releases/download/${RIPGREP_VERSION}/${asset}.tar.gz" \
    && echo "${digest}  /tmp/ripgrep.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/ripgrep.tar.gz -C /tmp \
    && install -m 0755 "/tmp/${asset}/rg" /usr/local/bin/rg \
    && rm -rf /tmp/ripgrep.tar.gz "/tmp/${asset}"

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
