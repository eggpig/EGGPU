# 静态 native 计时协议修订

协议 `native-host-csr-static-all4-public-return-v2` 对不可变 E1/E2 的 BFS/SSSP/WCC/PR 与 P=1/2/4/8 统一使用 `Session.native_static`。当前 runtime 源码仍是 71ea…，源码内容未改。公共 `Session.query` 返回完整 owning host result 后停止 query 计时，随后执行保留的 runner profile/device/encoding/PR 审核和 JSON 转换。公共 API 内的预算、资源收尾、ownership 检查均保留在时钟内。

这项修订明确改变旧协议的测量终点；旧收据原样保留，旧/新 protocol 的时间不得混合。62 项原始 CPU 测试通过只证明该 CPU 检查范围，公开投影另有 `public_cpu_receipt.json`。本包没有新增 GPU 性能声明；source-bound GPU 验证与完整输出检查由实际执行计划记录。

先用当前公开 runner 生成从未执行过的 71ea 计划，再运行：

```sh
python -B artifact/icde/campaign/native_static_measurement/prepare_plan.py \
  --plan /your/workspace/parent-plan.json \
  --output /your/workspace/amended-plan.json
```

脚本核对源身份和 parent worker 字节，不提交作业，不覆盖已有计划或测量。设置 `EGGPU_BENCH_CPU_SET` 为当地可用的 16 个不同 CPU；记录明确的 GPU、输入、构建与放置身份。公开路径投影及完整新文件哈希见 `projection.json` 和 `freeze.json`；原始冻结记录保存在 `upstream_freeze.json`，不用于直接授权当地构建。
