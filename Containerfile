# The gufo binary for AMD Strix Halo (gfx1151) with the ROCm runtime it was
# built against.
#
#   podman build -t rusty-gufo .
#   podman run --rm --device /dev/kfd --device /dev/dri --group-add keep-groups \
#     -p 127.0.0.1:8080:8080 -v "$HOME/models:/models:ro" \
#     -v rusty-gufo-cache:/root/.cache \
#     rusty-gufo serve --host 0.0.0.0 --port 8080 llm --model /models/<file>.gguf
#
# The cache volume keeps the code objects that ROCm's comgr links on the first
# load, which a new container would otherwise link again.
#
# ROCm comes from AMD's package repository in the qualified release that
# src/hip/build.rs checks, installed at the same paths as on a host. The build
# and runtime stages share one Ubuntu release: gufo opens FFmpeg by the
# versioned file names the build found.

ARG UBUNTU=26.04
ARG ROCM_RELEASE=10.1.0
ARG ROCM_PACKAGE_REVISION=3

FROM docker.io/library/ubuntu:${UBUNTU} AS rocm
ARG UBUNTU
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl gpg \
 && install -d /etc/apt/keyrings \
 && curl -fsSL https://stable.repo.amd.com/rocm/gpg/packages.gpg -o /tmp/amdrocm.key \
 && if grep -q 'BEGIN PGP' /tmp/amdrocm.key; then \
      gpg --dearmor < /tmp/amdrocm.key > /etc/apt/keyrings/amdrocm.gpg; \
    else \
      cp /tmp/amdrocm.key /etc/apt/keyrings/amdrocm.gpg; \
    fi \
 && rm /tmp/amdrocm.key \
 && printf '%s\n' 'Types: deb' \
      "URIs: https://stable.repo.amd.com/rocm/core/packages/ubuntu$(echo "$UBUNTU" | tr -d .)/" \
      'Suites: stable' 'Components: main' 'Architectures: amd64' \
      'Signed-By: /etc/apt/keyrings/amdrocm.gpg' \
      > /etc/apt/sources.list.d/amdrocm.sources \
 && rm -rf /var/lib/apt/lists/*

FROM rocm AS build
ARG ROCM_RELEASE
ARG ROCM_PACKAGE_REVISION
ARG GUFO_RELEASE_VERSION=
ARG GUFO_REVISION=
RUN series=${ROCM_RELEASE%.*} version=${ROCM_RELEASE}-${ROCM_PACKAGE_REVISION} \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential pkg-config \
      libavformat-dev libavcodec-dev libavutil-dev libswscale-dev \
      amdrocm-base${series}=${version} amdrocm-llvm${series}=${version} \
      amdrocm-runtime-dev${series}=${version} amdrocm-blas-dev${series}=${version} \
      amdrocm-hipblas-common-dev${series}=${version} amdrocm-ccl-dev${series}=${version} \
      amdrocm-blas${series}-gfx1151=${version} \
 && rm -rf /var/lib/apt/lists/*
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
RUN curl -fsSL https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain none --no-modify-path
WORKDIR /src
COPY . .
# The toolchain of rust-toolchain.toml; the caches keep crates and kernel
# objects between builds. The two GUFO_ build arguments reach the build script
# as environment variables.
RUN --mount=type=cache,target=/opt/cargo/registry \
    --mount=type=cache,target=/src/target \
    rustup toolchain install \
 && ROCM_PATH=/opt/rocm/core-${ROCM_RELEASE%.*} \
      cargo build --release --locked -p gufo \
 && install -m 755 target/release/gufo /usr/local/bin/gufo

FROM rocm AS runtime
ARG ROCM_RELEASE
ARG ROCM_PACKAGE_REVISION
RUN series=${ROCM_RELEASE%.*} version=${ROCM_RELEASE}-${ROCM_PACKAGE_REVISION} \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      libavformat62 libavcodec62 libavutil60 libswscale9 \
      amdrocm-base${series}=${version} amdrocm-runtime${series}=${version} \
      amdrocm-blas-host${series}=${version} amdrocm-blas${series}-gfx1151=${version} \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /usr/local/bin/gufo /usr/local/bin/gufo
# The license texts the binary's notices require (MIT, Apache-2.0 and the
# third-party licenses), where Debian-style images keep them.
COPY --from=build /src/LICENSE-MIT /src/LICENSE-APACHE /src/NOTICE /src/THIRD_PARTY_NOTICES.md /usr/share/doc/gufo/
COPY --from=build /src/licenses /usr/share/doc/gufo/licenses
RUN ln -s gufo /usr/local/bin/gufo-server
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/gufo"]
CMD ["--help"]
