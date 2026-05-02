#!/usr/bin/env python3
"""
从 docs 目录的动手实战文档中提取代码，创建对应的 Python 文件到 libs 目录
"""

import os
import re
from pathlib import Path


def extract_code_blocks(md_content):
    """
    从 markdown 内容中提取所有代码块
    返回列表：[(filename, code_content), ...]
    """
    # 匹配代码块：```python ... ```
    pattern = re.compile(
        r'```python\s*(?:#\s*(\S+\.py))?\s*\n(.*?)```',
        re.DOTALL
    )
    
    blocks = []
    for match in pattern.finditer(md_content):
        filename = match.group(1)
        code = match.group(2).strip()
        
        if filename:
            blocks.append((filename, code))
        else:
            # 如果没有指定文件名，尝试从代码中查找
            lines = code.split('\n')
            for line in lines[:5]:
                if line.startswith('#') and '.py' in line:
                    # 提取文件名
                    match_fn = re.search(r'(\S+\.py)', line)
                    if match_fn:
                        filename = match_fn.group(1)
                        break
            blocks.append((filename or "unknown.py", code))
    
    return blocks


def process_hands_on_files(docs_root, libs_root):
    """
    处理所有动手实战文件
    """
    # 找到所有动手实战相关的文件
    hands_on_files = []
    for root, dirs, files in os.walk(docs_root):
        for file in files:
            if file.endswith('.md') and '动手' in file:
                hands_on_files.append(Path(root) / file)
    
    print(f"找到 {len(hands_on_files)} 个动手实战文档")
    
    for md_file in hands_on_files:
        print(f"\n处理: {md_file}")
        
        # 读取文档内容
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取代码块
        code_blocks = extract_code_blocks(content)
        
        if not code_blocks:
            print("  警告：未找到代码块")
            continue
        
        # 确定目标目录结构
        # 从文件路径中提取章节信息
        relative_path = md_file.relative_to(docs_root)
        
        # 章节目录：去掉文件名，保留路径部分
        chapter_path = relative_path.parent
        
        # 实战标题：从文件名中提取（去掉编号和扩展名）
        title = md_file.stem
        # 清理标题中的特殊字符
        clean_title = re.sub(r'[【】\[\]()<>:"/\\|?*]', '_', title)
        clean_title = re.sub(r'_{2,}', '_', clean_title).strip('_')
        
        # 创建 libs 下的目标目录
        target_dir = libs_root / chapter_path / clean_title
        target_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  创建目录: {target_dir}")
        
        # 写入代码文件
        for filename, code in code_blocks:
            # 确保文件名安全
            safe_filename = re.sub(r'[【】\[\]()<>:"/\\|?*]', '_', filename)
            target_file = target_dir / safe_filename
            
            with open(target_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            print(f"    ✓ {safe_filename}")


if __name__ == '__main__':
    project_root = Path(__file__).parent
    docs_root = project_root / 'docs'
    libs_root = project_root / 'libs'
    
    if not docs_root.exists():
        print(f"错误：找不到 docs 目录: {docs_root}")
        exit(1)
    
    # 创建 libs 目录
    libs_root.mkdir(exist_ok=True)
    
    process_hands_on_files(docs_root, libs_root)
    
    print("\n✅ 处理完成！")
