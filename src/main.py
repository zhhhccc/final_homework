from m1 import operating_m1
from m2 import operating_m2
from m3 import operating_m3
from m4 import operating_m4

def main():
    print("=" * 40)
    print("  NYC Taxi 数据分析系统")
    print("=" * 40)
    print("  1 - M1: 数据质量报告与清洗")
    print("  2 - M2: 可视化分析（需求/区域/车费/支付）")
    print("  3 - M3: 机器学习预测（NN + 随机森林）")
    print("  4 - M4: 自然语言问答系统")
    print("  0 - 退出")
    print("=" * 40)

    choice = input("请选择模块 (0-4): ").strip()

    if choice == "1":
        operating_m1()
    elif choice == "2":
        operating_m2()
    elif choice == "3":
        operating_m3()
    elif choice == "4":
        operating_m4()
    elif choice == "0":
        print("退出系统。")
    else:
        print("无效输入，请输入 0-4。")

if __name__ == "__main__":
    main()
