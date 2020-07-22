FROM ubuntu:focal

RUN apt-get update

RUN apt-get install wget perl -y

RUN wget http://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz

RUN mkdir install-tl && tar -xf install-tl-unx.tar.gz -C install-tl --strip-components 1

WORKDIR install-tl

RUN yes "i" | perl install-tl

WORKDIR /

ENV PATH="/usr/local/texlive/2020/bin/x86_64-linux:${PATH}"