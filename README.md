## Readme

We did three different experiments using this framework. 

- Running the **original code**: you may access them in **/original_codes/python** to get slightly changed gpnn codes which can run on python 3.10 + pytorch2, but these codes may use plenty of old torch properties(although, still available). 
- Running the **rewrite code**: just running **main.py**, you may have to change the dataset you want to use.
- Running the **re-implement code**: you can access them in **/re-implement** folder. you may have to move them to the root folder to solve import problems, and there should be no other modifications to make.

we also provide .ipynb files with 

`!python main.py`

to show you what our terminal show in re-implement codes.

My codes were running on dsw-registry-vpc.cn-shanghai.cr.aliyuncs.com/pai/modelscope:1.20.0-pytorch2.4.0-gpu-py310-cu124-ubuntu22.04 and it runs fine to me. running codes are slightly changed as provided in this repo, you may wish to access the docker image through **crpi-t7horkw84s20et06.cn-shanghai.personal.cr.aliyuncs.com/gpnn/gpnn4py3**.

You may also interested in our other parts of codes, which you may request corresponding team member to pull requests or send e-mails to get their codes if needed. 

The code document(Chinese Version) could be accessed in code_document_cn.md.
