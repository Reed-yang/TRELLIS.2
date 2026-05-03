| bucket | ms/step | % |
|---|---|---|
| nccl | 693.6 | 40.5% |
| eltwise | 425.8 | 24.8% |
| gemm | 275.0 | 16.0% |
| other | 109.5 | 6.4% |
| flash_attn | 97.7 | 5.7% |
| reduce | 79.8 | 4.7% |
| layernorm | 27.9 | 1.6% |
| memcpy | 3.2 | 0.2% |
| optimizer | 1.8 | 0.1% |
| **TOTAL** | **1714.2** | **100%** |

comm_hidden_ratio: 0.686
compute_stream_busy_ms_per_step: 1020.7
comm_stream_busy_ms_per_step: 693.6