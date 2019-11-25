# FIXME rebuild for new release? I think I need explicitly pull new release.
# FROM tensorflow/tensorflow:latest-gpu-py3
FROM tensorflow/tensorflow:1.15.0-gpu-py3

RUN apt-get update -y
RUN apt-get install -y apt-utils
RUN apt-get install -y build-essential pkg-config cmake autoconf
RUN apt-get install -y git cloc
RUN apt-get install -y emacs
RUN apt-get install -y wget
RUN apt-get install -y locate

RUN apt-get install -y openbox
RUN apt-get install -y tigervnc-standalone-server
RUN apt-get install -y xterm
RUN apt-get install -y openssh-server
RUN apt-get install -y sudo
RUN apt-get install -y rxvt-unicode
RUN apt-get install -y tmux htop silversearcher-ag

# For showing iamges in emacs
RUN apt-get install -y imagemagick

# TODO I might need a browser
# RUN apt-get install -y chromium-browser
# FIXME chromium does not work inside container, firefox does
# RUN apt-get install -y firefox

# for elpy RPC
RUN apt-get install -y python3-venv

# install julia from source, apt does not have one
ENV JULIA_PATH /usr/local/julia
ENV PATH $JULIA_PATH/bin:$PATH
RUN curl -fL -o julia.tar.gz\
        "https://julialang-s3.julialang.org/bin/linux/x64/1.2/julia-1.2.0-linux-x86_64.tar.gz"
RUN mkdir "$JULIA_PATH"
RUN tar -xzf julia.tar.gz -C "$JULIA_PATH" --strip-components 1

# It looks like julia is not able to find the libraries. A closer look, many
# library files do not have the libxxx.so file, but libxxx.so.7. Thus I'll need
# to create the symlinks myself.
#
# Otherwise, you will have the error "CuArrays.jl failed to initialize and will
# be unavailable".  To see which library is needed, run
#
# env JULIA_CUDA_VERBOSE=true julia
# and using CuArrays

RUN ln -sf\
        /usr/lib/x86_64-linux-gnu/libcudnn.so.7\
        /usr/lib/x86_64-linux-gnu/libcudnn.so
RUN ln -sf\
        /usr/local/cuda-10.0/targets/x86_64-linux/lib/libcusparse.so.10.0\
        /usr/local/cuda-10.0/targets/x86_64-linux/lib/libcusparse.so
RUN ln -sf\
        /usr/local/cuda-10.0/targets/x86_64-linux/lib/libcublas.so.10.0\
        /usr/local/cuda-10.0/targets/x86_64-linux/lib/libcublas.so
RUN ln -sf\
        /usr/local/cuda-10.0/targets/x86_64-linux/lib/libcusolver.so.10.0\
        /usr/local/cuda-10.0/targets/x86_64-linux/lib/libcusolver.so

ENV DOCKER_ENVIRONMENT "set"

RUN useradd -m -s /bin/bash user &&\
        echo "user:user" | chpasswd &&\
        usermod -aG sudo user
RUN echo "root:root" | chpasswd

USER user

WORKDIR /home/user
RUN git clone https://github.com/lihebi/emacs.d .emacs.d
RUN git clone https://github.com/lihebi/dothebi .hebi &&\
        sh .hebi/install.sh &&\
        sh .hebi/setup-git.sh

# NOTE: do not install keras
RUN pip install --user matplotlib seaborn cleverhans ipython foolbox pillow tqdm

RUN mkdir git
RUN cd git && git clone https://github.com/lihebi/AdvAE
RUN cd git && git clone https://github.com/MadryLab/mnist_challenge
RUN cd git && git clone https://github.com/MadryLab/cifar10_challenge

RUN mkdir .vnc
ADD xstartup .vnc/xstartup
ADD Xresources .Xresources

RUN echo "vnc" | vncpasswd -f > .vnc/passwd &&\
        chmod 600 .vnc/passwd

ADD startvnc.sh startvnc.sh
CMD ["/home/user/startvnc.sh"]

USER root
RUN sudo chown -R user:user /home/user
USER user


