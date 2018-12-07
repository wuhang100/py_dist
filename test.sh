#!/bin/bash
python /home/wuhang/LSTM_code/split.py
start=`date +%s`
scp /home/wuhang/LSTM_code/inputdata11.npy Data1:/home/wuhang/LSTM_code
scp /home/wuhang/LSTM_code/outdata11.npy Data1:/home/wuhang/LSTM_code
ssh -tt Data1 "/home/wuhang/anaconda2/bin/python /home/wuhang/LSTM_code/train2.py;exit" &
python /home/wuhang/LSTM_code/train.py &
wait
end=`date +%s`
echo "TIME:`expr $end - $start`"
scp Data1:/home/wuhang/LSTM_code/model02.h5 /home/wuhang/LSTM_code
python /home/wuhang/LSTM_code/getdata.py
