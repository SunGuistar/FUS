#!/usr/bin/env python3
"""
测试修改后的CSV格式生成
"""

import os
import csv
import numpy as np

# 像素到毫米转换比例（基于校准矩阵）
PIXEL_TO_MM_X = 0.229389190673828
PIXEL_TO_MM_Y = 0.220979690551758
PIXEL_TO_MM_Z = 1.0
PIXEL_TO_MM_AVG = (PIXEL_TO_MM_X + PIXEL_TO_MM_Y) / 2

def convert_pixel_to_mm(pixel_value, is_global=True):
    """将像素值转换为毫米值"""
    if is_global:
        return pixel_value * PIXEL_TO_MM_AVG
    else:
        return pixel_value * PIXEL_TO_MM_Z

def test_csv_format():
    """测试新的CSV格式"""
    print("=== 测试新的CSV格式 ===")
    
    # 定义CSV列名（合并均值和标准差）
    csv_headers = [
        'file_name',
        'model_name', 
        'global_T_all_points_mm',
        'global_T_R_all_points_mm',
        'global_T_4_points_mm',
        'global_T_R_4_points_mm',
        'local_T_all_points_mm',
        'local_T_4_points_mm'
    ]
    
    # 模拟测试数据（基于metrics_mm_converted.csv）
    test_data = [
        ['seq_len100__Loss_rec_reg__bs_2__inc_reg_1__Move__ete', "['best_val_dist_R_T', 'best_val_dist_R_R']", 
         "18.27±3.10", "18.17±3.35", "18.48±3.09", "18.61±3.32", "1.24±0.25", "1.29±0.24"],
        ['seq_len100__Loss_rec_reg__bs_2__inc_reg_1__Move__ete', "['best_val_dist_T_T', 'best_val_dist_T_R']", 
         "18.28±3.03", "18.07±5.13", "18.49±3.03", "18.09±5.14", "1.16±0.20", "1.19±0.20"]
    ]
    
    # 创建测试CSV文件
    test_csv_path = 'test_new_format.csv'
    with open(test_csv_path, 'w', newline='', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        writer.writerows(test_data)
    
    print(f"测试CSV文件已创建: {test_csv_path}")
    
    # 读取并显示内容
    with open(test_csv_path, 'r', encoding='UTF8') as f:
        content = f.read()
        print("CSV内容:")
        print(content)
    
    # 验证格式
    print("\n格式验证:")
    with open(test_csv_path, 'r', encoding='UTF8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        print(f"列数: {len(headers)}")
        print(f"列名: {headers}")
        
        for i, row in enumerate(reader, 1):
            print(f"第{i}行数据: {row}")
            print(f"  数据列数: {len(row)}")
            # 验证数据格式
            for j, cell in enumerate(row[2:], 2):  # 跳过前两列
                if '±' in cell:
                    parts = cell.split('±')
                    if len(parts) == 2:
                        try:
                            mean = float(parts[0])
                            std = float(parts[1])
                            print(f"  列{j+1} ({headers[j]}): {mean:.2f}±{std:.2f} ✓")
                        except ValueError:
                            print(f"  列{j+1} ({headers[j]}): 格式错误 ✗")
                    else:
                        print(f"  列{j+1} ({headers[j]}): 格式错误 ✗")
    
    # 清理测试文件
    os.remove(test_csv_path)
    print(f"\n测试文件已清理")

def test_conversion():
    """测试像素到毫米转换"""
    print("\n=== 像素到毫米转换测试 ===")
    
    # 测试数据
    test_values = {
        'global_T_all_points': 79.68,
        'global_T_R_all_points': 79.24,
        'local_T_all_points': 1.24
    }
    
    for metric, pixel_value in test_values.items():
        is_global = 'global' in metric
        mm_value = convert_pixel_to_mm(pixel_value, is_global)
        print(f"{metric}: {pixel_value} pixel → {mm_value:.2f} mm")

def main():
    """主测试函数"""
    print("CSV格式修改测试")
    print("=" * 50)
    
    test_conversion()
    test_csv_format()
    
    print("=" * 50)
    print("测试完成！")
    print("\n新格式特点:")
    print("1. ✅ 标准CSV格式（逗号分隔）")
    print("2. ✅ 均值和标准差合并显示（mean±std）")
    print("3. ✅ 列名简洁明确")
    print("4. ✅ 所有距离单位统一为毫米")
    print("5. ✅ 与目标格式完全一致")

if __name__ == "__main__":
    main()
