# WCC 宽索引实验候选：6cce

本目录是相对已公开 d84 主源码的 20 文件增量，包含 19 份 C++/CUDA 相关改动及 Python 公开动态入口的宽 offset 门。它不替换主 `Easy-Graph/`。候选 SHA-256 为 `6cce329e95f6fb10831d698845677579cfd05bfbf50c899132d03ed20cecab81`；完整文件身份与基底记录在 `source_patch.json`。没有分发机器专用二进制。

WCC 的本地 CSR offset 扩展为 int64，顶点 ID 仍为 int32；其他函数保持已有数值域拒绝。默认仍为 auto + forest_coo；label/forest 替代路径和强制宽度用于资格覆盖，不是按图选择最快分支。126 项候选 CPU 检查、260 项原核心检查及 host 等价检查的历史本机通过，只证明 CPU 范围。80 项组合 GPU 资格已提交，完成前不能宣称 TW 可运行、性能更快或主版本已验证。

先从仓库根目录在新目录重建候选，命令不调用 CUDA：

```sh
python -B artifact/icde/experimental/native_wcc_wide_v2/materialize.py --output /your/new-candidate
EGGPU_BASE_SOURCE="$PWD/Easy-Graph" PYTHONPATH=/your/new-candidate/Easy-Graph CUDA_VISIBLE_DEVICES='' \
python -B -m pytest -q -p no:cacheprovider --confcutdir=/your/new-candidate \
  /your/new-candidate/test_wide_offsets.py /your/new-candidate/test_wide_adapter.py \
  /your/new-candidate/test_dynamic_wide_gate.py
```

编译应对新目录的 `Easy-Graph` 使用项目现有 CMake/构建说明和调用者 CUDA、编译器环境，记录完整命令、源码与 DSO 哈希。历史 DSO `2ad2d06504a21b1e5e07248170506a73da4508a13126aa87852a179ba8d96d63` 是从父候选复用的同字节二进制，不是 d84 DSO，也不是 GPU 资格证明。新编译 DSO 放在候选工作目录 `native/` 后，`test_wide_compiled.py` 才能执行真实加载/clone CPU 检查。

`runner/` 保留真实 qualification worker 和数值域入口。执行 GPU 前须在自己的机器重建输入、设备、DSO、源码、解释器和资源 binding，先完成完整 tiny/目标资格，再提交正式测量。公开包没有自动绑定本地机器或自动启动 GPU 的入口。
