def count_in_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()  # 保留所有行，包含换行符（可能已去除）
    
    # 总统计
    total_digits = 0
    total_left = 0
    total_right = 0
    
    print("===== 按行统计括号数量 =====")
    for i, line in enumerate(lines, 1):
        # 去除行尾换行符，避免干扰
        clean_line = line.rstrip('\n').rstrip('\r')
        
        left = clean_line.count('(')
        right = clean_line.count(')')
        
        total_digits += sum(1 for ch in clean_line if ch.isdigit())
        total_left += left
        total_right += right
        
        # 打印该行的括号数量 (若行不为空，或括号数大于0时打印，可根据需求调整)
        if left > 0 or right > 0:   # 仅打印包含括号的行，减少输出；若想打印所有行，去掉此条件
            print(f"第{i:4d}行: '(' {left:3d}   ')' {right:3d}")
    
    print("\n===== 总体统计 =====")
    print(f"数字字符数量: {total_digits}")
    print(f"'(' 数量: {total_left}")
    print(f"')' 数量: {total_right}")
    print(f"总行数: {len(lines)}")

if __name__ == "__main__":
    path = input("请输入txt文件路径: ").strip()
    count_in_file(path)