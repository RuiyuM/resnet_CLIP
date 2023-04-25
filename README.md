cd to data dir and use the following command to download: wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
 or curl -O http://cs231n.stanford.edu/tiny-imagenet-200.zip.
then use: unzip tiny-imagenet-200.zip.
then use nohup python execute_commands.py. it will run the code every 30 mins.
or use nohup python exe_para.py. it will run code in 8 separate GPU and once a program finished it will start new.
The generate.py is the code which can generate the 
required baseline.txt file. 

The current baseline do not include the original paper's method,
because My desktop has conda envr problem and once I fix the problem
I will test the AL_temp method and update the code.
