FROM python:3.10-slim-bookworm

RUN apt-get update && apt install -y openssh-server zip vim  \
    # Clean up to keep the image small
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /var/run/sshd

WORKDIR /workdir

EXPOSE 22

CMD ["git", "clone", "https://github.com/anthonyceponis/mvp-depth-refinement.git"]
