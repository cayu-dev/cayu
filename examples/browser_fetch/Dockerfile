ARG PYTHON_IMAGE=python:3.13-slim-bookworm@sha256:00faa2debb87529f9f0764e9491d8ba400a3678976616c3bd7cb193745ac20d1
FROM ${PYTHON_IMAGE} AS chromium-sandbox-helper

ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

RUN python -m pip install --no-cache-dir "playwright==1.62.0" \
    && python -m playwright install --no-shell chromium \
    && sandbox_path="$(find /ms-playwright/chromium-* -type f -name chrome_sandbox -print -quit)" \
    && test -n "${sandbox_path}" \
    && install --owner=root --group=root --mode=4755 "${sandbox_path}" /chrome-sandbox \
    && test -u /chrome-sandbox

FROM ${PYTHON_IMAGE}

LABEL org.opencontainers.image.title="Cayu browser inspection worker" \
      org.opencontainers.image.version="6-playwright-1.62.0" \
      dev.cayu.browser-fetch.protocol="cayu.browser-fetch.v4" \
      dev.cayu.browser-fetch.worker="4" \
      dev.cayu.browser-session.protocol="cayu.browser-session.v2" \
      dev.cayu.browser-session.worker="6" \
      dev.cayu.browser-fetch.playwright="1.62.0"

USER root

ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

RUN python -m pip install --no-cache-dir "playwright==1.62.0" \
    && python -m playwright install --with-deps chromium-headless-shell \
    && apt-get update \
    && apt-get install --yes --no-install-recommends libnss3-tools \
    && groupadd --gid 1000 pwuser \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash pwuser \
    && chmod -R a+rX /ms-playwright \
    && rm -rf /var/lib/apt/lists/*

COPY --from=chromium-sandbox-helper /chrome-sandbox /tmp/chrome-sandbox

RUN shell_path="$(find /ms-playwright/chromium_headless_shell-* -type f \( -name headless_shell -o -name chrome-headless-shell \) -print -quit)" \
    && test -n "${shell_path}" \
    && test -x "${shell_path}" \
    && sandbox_target="$(dirname "${shell_path}")/chrome-sandbox" \
    && install --owner=root --group=root --mode=4755 /tmp/chrome-sandbox "${sandbox_target}" \
    && rm /tmp/chrome-sandbox \
    && test -u "${sandbox_target}"

COPY --chown=root:root src/cayu/tools/_browser_guest.py /opt/cayu-browser/worker.py

RUN chmod 0555 /opt/cayu-browser/worker.py \
    && python -m py_compile /opt/cayu-browser/worker.py \
    && python -c "import importlib.metadata; assert importlib.metadata.version('playwright') == '1.62.0'"

USER pwuser
WORKDIR /workspace
