# ===== Base Rocq + Rocq-lsp =====
FROM rocq/rocq-prover:9.0

SHELL ["/bin/bash", "-lc"]
ENV OPAMYES=1 OPAMCOLOR=never

# Use the non-root user provided by the base image
USER rocq
WORKDIR /home/rocq

# Create a simple, named switch and install tooling + MathComp
RUN if [ ! -d "$HOME/.opam" ]; then opam init --bare --disable-sandboxing; fi && \
    opam repo add coq-released https://coq.inria.fr/opam/released && \
    opam update && \
    opam upgrade && \
    opam unpin ocamlfind && \
    opam pin coq-lsp 0.2.5+9.0 && \
    opam install -y logs lwt coq-lsp

