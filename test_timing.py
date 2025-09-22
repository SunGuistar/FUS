#!/usr/bin/env python3
"""
测试时间显示功能的简单脚本
"""

import time
from datetime import datetime, timedelta

def test_timing_display():
    """测试时间显示功能"""
    print("=== 测试时间显示功能 ===")
    
    # 模拟epoch开始
    epoch_start_time = time.time()
    print(f"\n=== Epoch 0 Started ===")
    
    # 模拟训练
    print("开始训练...")
    time.sleep(2)  # 模拟训练时间
    print('[Rec - Epoch 0] train-loss-rec=0.123, train-dist=0.456')
    
    # 模拟验证
    print("开始验证...")
    time.sleep(1)  # 模拟验证时间
    print('[Rec - Epoch 0] val-loss-rec=0.234, val-dist=0.567')
    
    # 模拟epoch结束
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    epoch_duration_str = str(timedelta(seconds=int(epoch_duration)))
    print(f"=== Epoch 0 Completed in {epoch_duration_str} ===")
    
    print("\n✅ 时间显示功能测试完成！")

if __name__ == "__main__":
    test_timing_display()
