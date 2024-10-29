import os  
  
def rename_txt_files(directory):   
    if not os.path.isdir(directory):  
        raise FileNotFoundError(f"Not exist")  
       
    for filename in os.listdir(directory):  
        if filename.endswith('.txt'):  
            base, ext = os.path.splitext(filename)  
            try:  
                num = int(base)  
                new_base = f"{num:06d}"   
                new_filename = new_base + ext  
                old_filepath = os.path.join(directory, filename)  
                new_filepath = os.path.join(directory, new_filename)   
                os.rename(old_filepath, new_filepath)  
                print(f"Renamed: {filename} -> {new_filename}")  
            except ValueError:   
                print(f"Skipped: {filename} (not a valid integer)")  

rename_txt_files("/root/autodl-tmp/StableMoFusion/activity/train/text")