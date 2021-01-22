#!/bin/bash
#load the cuda environment
source /opt/asn/etc/asn-bash-profiles-special/modules.sh
module load cuda/10.1.168
#run your software here
./main a_3_3.mtx b_3_2.mtx
./main a_32_32.mtx b_32_32.mtx
./main a_1024_1024.mtx b_1024_1024.mtx
./main a_2048_2048.mtx b_2048_2048.mtx


