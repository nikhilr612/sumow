"""sumow — JAX/Equinox implementation of super weight analysis for LLMs."""

__version__ = "0.1.0"

from sumow.config import (
    SUPER_WEIGHT_DIRECTORY,
    ClipMethod,
    EvalConfig,
    IdentifyConfig,
    ModelConfig,
    QuantizationConfig,
)
from sumow.eval import cross_entropy_loss, perplexity, perplexity_from_loss
from sumow.identify import (
    LayerActivationStats,
    SuperWeight,
    compute_layer_stats,
    detect_spikes,
    identify_super_weights,
)
from sumow.model import (
    LlamaAttention,
    LlamaBlock,
    LlamaMLP,
    LlamaModel,
    RMSNorm,
    RotaryEmbedding,
    TransformerConfig,
    forward_jit,
    load_weights_into_model,
    make_hf_weights_dict,
)
from sumow.model_io import (
    extract_down_proj_weights,
    get_super_weight_values,
    load_model_weights,
    load_safetensors,
)
from sumow.quantize import (
    QuantizeResult,
    quantize_activation_sa_aware,
    quantize_dequantize_blockwise,
    quantize_weight_sw_aware,
)
from sumow.benchmark import (
    BenchmarkReport,
    BenchmarkResult,
    PAPER_CONFIGS,
    QuantConfig,
    run_benchmark,
    run_benchmark_with_identification,
)
