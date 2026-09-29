# Browser variant of the first-party Lambda MicroVM image. Build it from the
# context written by `cayu lambda-microvm sidecar export --browser DESTINATION`,
# which places the sidecar under lambda_microvm_sidecar/ and this Cayu release's
# browser worker sources under browser-worker/.
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

# Chromium's shared libraries and NSS tools. Playwright's --with-deps only
# supports apt, so the dependency set is listed for dnf.
RUN dnf install -y \
        alsa-lib \
        at-spi2-atk \
        at-spi2-core \
        atk \
        cairo \
        cups-libs \
        dbus-libs \
        expat \
        fontconfig \
        libX11 \
        libXcomposite \
        libXdamage \
        libXext \
        libXfixes \
        libXrandr \
        libdrm \
        libxcb \
        libxkbcommon \
        libxshmfence \
        mesa-libgbm \
        nspr \
        nss \
        nss-tools \
        pango \
    && dnf clean all

# The worker command is `/usr/local/bin/python -I /opt/cayu-browser/worker.py`
# and it launches Chromium from /ms-playwright. Chromium keeps its own sandbox:
# agent commands run with no_new_privs, so it uses unprivileged user namespaces
# rather than a setuid helper. Amazon Linux's Python leaves /usr/local
# site-packages off sys.path in isolated mode (-I), so the worker's packages
# are installed under /usr.
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN python3.11 -m pip install --no-cache-dir --prefix=/usr \
        "playwright==1.62.0" "websockets==17.0.1" \
    && python3.11 -m playwright install chromium-headless-shell \
    && chmod -R a+rX,go-w /ms-playwright \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && fc-cache -f

COPY browser-worker/ /opt/cayu-browser/
RUN chown -R root:root /opt/cayu-browser \
    && chmod 0755 /opt/cayu-browser \
    && chmod 0444 /opt/cayu-browser/*.py \
    && chmod 0555 /opt/cayu-browser/worker.py \
    && /usr/local/bin/python -I -c "import importlib.metadata as m; assert m.version('playwright') == '1.62.0'; assert m.version('websockets') == '17.0.1'" \
    && test -x /usr/bin/certutil

ENV CAYU_MICROVM_AGENT_UID=1000 \
    CAYU_MICROVM_AGENT_GID=1000 \
    CAYU_MICROVM_AGENT_NETNS=cayu-agent

WORKDIR /opt/cayu
COPY lambda_microvm_sidecar /opt/cayu/lambda_microvm_sidecar
RUN python3.11 -m pip install --no-cache-dir \
        -r /opt/cayu/lambda_microvm_sidecar/requirements.txt \
        "botocore>=1.43.44,<2" \
    && install -d -o 1000 -g 1000 /workspace

EXPOSE 8080
# Reads the browser files before the ready hook, then runs the sidecar
# entrypoint. Create the image with a ready-hook timeout of at least 300 seconds.
CMD ["bash", "/opt/cayu/lambda_microvm_sidecar/browser/start.sh"]
