import os
import zipfile

def create_zip(zip_filename, source_dir):
    """
    将项目打包为 ZIP 文件，自动忽略无用文件 (如 .git, __pycache__, 数据集缓存等)
    以极小化体积，方便上传至云服务器。
    """
    # 需要忽略的文件夹和文件后缀
    ignore_dirs = {
        '.git', '.vscode', '__pycache__', 'datasets', 'checkpoints', 
        'logs', '.venv', '.idea', 'image'
    }
    ignore_exts = {'.pyc', '.pth', '.pt', '.zip'}
    
    print(f"📦 开始打包项目到: {zip_filename} ...")
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            # 过滤掉不需要的文件夹
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            for file in files:
                # 过滤掉不需要的文件后缀
                if any(file.endswith(ext) for ext in ignore_exts):
                    continue
                    
                # 排除自身
                if file == zip_filename:
                    continue
                    
                file_path = os.path.join(root, file)
                # 计算相对路径，使 ZIP 包内的结构保持一致
                arcname = os.path.relpath(file_path, source_dir)
                
                zipf.write(file_path, arcname)
                print(f"  + {arcname}")
                
    print(f"\n✅ 打包完成！文件大小: {os.path.getsize(zip_filename) / (1024*1024):.2f} MB")
    print(f"👉 你现在可以将 {zip_filename} 上传到 AutoDL 云服务器了。")

if __name__ == "__main__":
    current_dir = os.getcwd()
    zip_name = "HiLo_Math_Extended.zip"
    create_zip(zip_name, current_dir)
