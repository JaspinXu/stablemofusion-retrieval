import os  
  
def find_multiline_txt_files(directory):  
    # 获取指定目录下的所有文件和文件夹  
    muti_line_files=[]
    for root, dirs, files in os.walk(directory):  
        for file in files:  
            # 检查文件扩展名是否为.txt  
            if file.endswith('.txt'):  
                file_path = os.path.join(root, file)  
                  
                # 打开文件并读取内容  
                with open(file_path, 'r', encoding='utf-8') as f:  
                    lines = f.readlines()  
                  
                # 如果文件有多行（即行数大于1），打印文件名  
                if len(lines) > 1:  
                    muti_line_files.append(file)  
    print(muti_line_files)
  
# 调用函数并传入要遍历的目录路径  
directory_path = '/root/autodl-tmp/StableMoFusion/act/text/'  # 替换为你的目录路径  
find_multiline_txt_files(directory_path)