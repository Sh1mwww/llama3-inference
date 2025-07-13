#!/bin/bash
# GSP错误修复脚本

echo "=== RTX 5080 GSP错误修复方案 ==="

echo "1. 检查当前GPU状态..."
nvidia-smi -q -d POWER,PERFORMANCE | grep -A 5 "Performance State\|Power"

echo -e "\n2. 尝试启用持久模式（需要sudo权限）..."
echo "sudo nvidia-smi -pm 1"

echo -e "\n3. 设置GPU性能模式为最大（防止进入深度休眠）..."
echo "sudo nvidia-smi -lgc 300,2000"  # 设置时钟范围

echo -e "\n4. 禁用GPU电源管理（临时解决方案）..."
echo "sudo bash -c 'echo 0 > /sys/bus/pci/devices/0000:02:00.0/power/runtime_enabled'"

echo -e "\n5. NVIDIA驱动参数调整..."
echo "添加到 /etc/modprobe.d/nvidia.conf："
echo "options nvidia NVreg_EnableGpuFirmware=0"
echo "options nvidia NVreg_EnableGpuFirmwareGSP=0"

echo -e "\n6. 检查系统日志中的GSP错误..."
echo "sudo dmesg | grep -i gsp | tail -10"

echo -e "\n=== 自动修复脚本 ==="
cat << 'EOF'
# 如果有sudo权限，运行以下命令：

# 启用持久模式
sudo nvidia-smi -pm 1

# 设置性能模式
sudo nvidia-smi -lgc 300,2000

# 禁用GSP固件（重启后生效）
echo 'options nvidia NVreg_EnableGpuFirmware=0' | sudo tee -a /etc/modprobe.d/nvidia.conf
echo 'options nvidia NVreg_EnableGpuFirmwareGSP=0' | sudo tee -a /etc/modprobe.d/nvidia.conf

# 重建initramfs
sudo update-initramfs -u

echo "请重启系统使修改生效"
EOF

echo -e "\n=== 临时缓解方案（无需重启） ==="
echo "如果无sudo权限，可以："
echo "1. 定期运行轻量级GPU任务保持唤醒"
echo "2. 使用我们项目中的GPU监控"
echo "3. 设置较小的内存限制避免触发GSP"