#!/bin/bash

tar zcvf back/archive-$(date +%m.%d.%y-%H.%M.%S).tar.gz trained tensorboard_logs/
