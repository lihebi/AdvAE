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


