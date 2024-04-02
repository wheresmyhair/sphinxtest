import os

# 指定要遍历的文件夹路径
folder_path = './'
find_content = '.footer-item'

# 用来存储包含特定字符串的文件的名称和路径
matching_files = []

# 递归遍历文件夹
for dirpath, dirnames, filenames in os.walk(folder_path):
    for filename in filenames:
        # 检查文件扩展名
        if filename.endswith('.js') or filename.endswith('.css'):
            # 构造完整的文件路径
            file_path = os.path.join(dirpath, filename)
            # 打开并读取文件内容
            with open(file_path, 'r', encoding='utf-8') as file:
                if find_content in file.read():
                    # 如果文件包含特定字符串，则添加到列表中
                    matching_files.append(file_path)

# 打印包含特定字符串的文件的名称和路径
for file_path in matching_files:
    print(file_path)

if not matching_files:
    print(f"没有找到包含 '{find_content}' 的文件。")
