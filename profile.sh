#!/bin/bash
# ================================
# 
# 用法示例:
#   ./profile.sh "Hello world"                    # 直接文本输入
#   ./profile.sh /path/to/prompts.txt             # 文件输入
#   ./profile.sh prompts.txt -l 256 -b 32 -v     # 指定参数
#   ./profile.sh prompts.txt --csv results.csv   # 导出CSV
#   ./profile.sh prompts.txt --runs 3 --topk 8   # 多次运行
# 
# ================================

set -e  # 遇到错误立即退出

# ==================== 配置参数 ====================
# 默认模型路径（可通过环境变量覆盖）
DEFAULT_MODEL="${LLAMA_MODEL_PATH:-/home/roger/.llama/checkpoints/Llama3.1-8B}"
DEFAULT_DEVICE="${LLAMA_DEVICE:-cuda}"
DEFAULT_BATCH_SIZE=32
DEFAULT_MAX_LEN=2048
DEFAULT_TOPK=4
DEFAULT_RUNS=1

# 颜色输出定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ==================== 帮助函数 ====================
show_help() {
    echo -e "${CYAN}LLaMA3 性能分析工具 v2.0${NC}"
    echo ""
    echo -e "${YELLOW}用法:${NC}"
    echo "  $0 <prompt_or_file> [options]"
    echo ""
    echo -e "${YELLOW}参数:${NC}"
    echo "  prompt_or_file      输入提示文本或包含提示的文件路径"
    echo ""
    echo -e "${YELLOW}选项:${NC}"
    echo "  -h, --help         显示此帮助信息"
    echo "  -m, --model PATH   模型路径 (默认: $DEFAULT_MODEL)"
    echo "  -d, --device DEV   设备类型 (默认: $DEFAULT_DEVICE)"
    echo "  -l, --length N     最大生成长度 (默认: $DEFAULT_MAX_LEN)"
    echo "  -b, --batch N      批处理大小 (默认: $DEFAULT_BATCH_SIZE)"
    echo "  -t, --topk N       GPU中保留的KV块数 (默认: $DEFAULT_TOPK)"
    echo "  -r, --runs N       运行次数 (默认: $DEFAULT_RUNS)"
    echo "  -c, --csv FILE     导出CSV结果到文件"
    echo "  -v, --verbose      启用详细日志输出"
    echo "  --dry-run          仅显示命令，不执行"
    echo ""
    echo -e "${YELLOW}环境变量:${NC}"
    echo "  LLAMA_MODEL_PATH   默认模型路径"
    echo "  LLAMA_DEVICE       默认设备类型"
    echo ""
    echo -e "${YELLOW}示例:${NC}"
    echo "  $0 \"Tell me a story\""
    echo "  $0 prompts.txt -l 256 -b 32 -v"
    echo "  $0 prompts.txt --csv results.csv --runs 3"
    echo "  $0 /path/to/prompts.txt -m /path/to/model -d cpu"
}

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${PURPLE}[DEBUG]${NC} $1"
    fi
}

# 验证文件是否存在
validate_file() {
    local file="$1"
    local description="$2"
    
    if [[ ! -f "$file" ]]; then
        log_error "$description 不存在: $file"
        exit 1
    fi
    
    if [[ ! -r "$file" ]]; then
        log_error "$description 不可读: $file"
        exit 1
    fi
}

# 验证目录是否存在
validate_directory() {
    local dir="$1"
    local description="$2"
    
    if [[ ! -d "$dir" ]]; then
        log_error "$description 不存在: $dir"
        exit 1
    fi
}

# 验证数字参数
validate_number() {
    local value="$1"
    local name="$2"
    local min_value="${3:-1}"
    
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        log_error "$name 必须是正整数: $value"
        exit 1
    fi
    
    if [[ "$value" -lt "$min_value" ]]; then
        log_error "$name 必须大于等于 $min_value: $value"
        exit 1
    fi
}

# 检查Python环境
check_python_env() {
    log_debug "检查Python环境..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安装或不在PATH中"
        exit 1
    fi
    
    local python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    log_debug "Python版本: $python_version"
    
    # 检查profile_pipeline.py是否存在
    local script_path="scripts/profile_pipeline.py"
    if [[ ! -f "$script_path" ]]; then
        log_error "分析脚本不存在: $script_path"
        exit 1
    fi
}

# 检测GPU可用性
check_gpu() {
    if [[ "$DEVICE" == "cuda" ]]; then
        log_debug "检查CUDA可用性..."
        
        if command -v nvidia-smi &> /dev/null; then
            local gpu_count=$(nvidia-smi --list-gpus | wc -l)
            if [[ "$gpu_count" -gt 0 ]]; then
                log_info "检测到 $gpu_count 个GPU设备"
                nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | while read line; do
                    log_debug "GPU: $line"
                done
            else
                log_warn "未检测到GPU设备，将使用CPU"
                DEVICE="cpu"
            fi
        else
            log_warn "nvidia-smi 不可用，将使用CPU"
            DEVICE="cpu"
        fi
    fi
}

# ==================== 参数解析 ====================
PROMPT_INPUT=""
MODEL_PATH="$DEFAULT_MODEL"
DEVICE="$DEFAULT_DEVICE"
MAX_LEN="$DEFAULT_MAX_LEN"
BATCH_SIZE="$DEFAULT_BATCH_SIZE"
TOPK="$DEFAULT_TOPK"
RUNS="$DEFAULT_RUNS"
CSV_OUTPUT=""
VERBOSE="false"
DRY_RUN="false"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        -l|--length)
            MAX_LEN="$2"
            shift 2
            ;;
        -b|--batch)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -t|--topk)
            TOPK="$2"
            shift 2
            ;;
        -r|--runs)
            RUNS="$2"
            shift 2
            ;;
        -c|--csv)
            CSV_OUTPUT="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE="true"
            shift
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        -*)
            log_error "未知选项: $1"
            show_help
            exit 1
            ;;
        *)
            if [[ -z "$PROMPT_INPUT" ]]; then
                PROMPT_INPUT="$1"
            else
                log_error "多余的参数: $1"
                show_help
                exit 1
            fi
            shift
            ;;
    esac
done

# ==================== 参数验证 ====================
log_debug "开始参数验证..."

# 检查必需参数
if [[ -z "$PROMPT_INPUT" ]]; then
    log_error "缺少输入参数"
    show_help
    exit 1
fi

# 验证模型路径
validate_directory "$MODEL_PATH" "模型目录"

# 验证数字参数
validate_number "$MAX_LEN" "生成长度" 1
validate_number "$BATCH_SIZE" "批处理大小" 1
validate_number "$TOPK" "TopK块数" 1
validate_number "$RUNS" "运行次数" 1

# 验证设备参数
if [[ "$DEVICE" != "cuda" && "$DEVICE" != "cpu" ]]; then
    log_error "无效的设备类型: $DEVICE (支持: cuda, cpu)"
    exit 1
fi

# ==================== 环境检查 ====================
log_info "🔍 检查运行环境..."
check_python_env
check_gpu

# ==================== 处理输入 ====================
log_debug "处理输入参数..."

PROMPT_ARG=""
if [[ -f "$PROMPT_INPUT" ]]; then
    validate_file "$PROMPT_INPUT" "提示文件"
    PROMPT_ARG="--prompt-file \"$PROMPT_INPUT\""
    log_info "📝 使用提示文件: $PROMPT_INPUT"
    
    # 显示文件信息
    line_count=$(wc -l < "$PROMPT_INPUT")
    log_debug "文件包含 $line_count 行"
else
    PROMPT_ARG="--prompt \"$PROMPT_INPUT\""
    log_info "📝 使用直接文本输入"
    log_debug "提示内容: ${PROMPT_INPUT:0:100}..."
fi

# ==================== 构建命令 ====================
log_debug "构建执行命令..."

PYTHON_CMD="python3 scripts/profile_pipeline.py"
PYTHON_CMD="$PYTHON_CMD --model-path \"$MODEL_PATH\""
PYTHON_CMD="$PYTHON_CMD --device \"$DEVICE\""
PYTHON_CMD="$PYTHON_CMD $PROMPT_ARG"
PYTHON_CMD="$PYTHON_CMD --max-gen-len $MAX_LEN"
PYTHON_CMD="$PYTHON_CMD --batch-size $BATCH_SIZE"
PYTHON_CMD="$PYTHON_CMD --topk-blk $TOPK"
PYTHON_CMD="$PYTHON_CMD --runs $RUNS"

if [[ "$VERBOSE" == "true" ]]; then
    PYTHON_CMD="$PYTHON_CMD --verbose"
fi

if [[ -n "$CSV_OUTPUT" ]]; then
    PYTHON_CMD="$PYTHON_CMD --csv \"$CSV_OUTPUT\""
    log_info "📊 CSV结果将保存到: $CSV_OUTPUT"
fi

# ==================== 显示配置 ====================
echo ""
echo -e "${CYAN}===================== 运行配置 =====================${NC}"
echo -e "${BLUE}模型路径:${NC} $MODEL_PATH"
echo -e "${BLUE}设备类型:${NC} $DEVICE"
echo -e "${BLUE}生成长度:${NC} $MAX_LEN"
echo -e "${BLUE}批处理大小:${NC} $BATCH_SIZE"
echo -e "${BLUE}TopK块数:${NC} $TOPK"
echo -e "${BLUE}运行次数:${NC} $RUNS"
if [[ -n "$CSV_OUTPUT" ]]; then
    echo -e "${BLUE}CSV输出:${NC} $CSV_OUTPUT"
fi
echo -e "${BLUE}详细日志:${NC} $VERBOSE"
echo -e "${CYAN}================================================${NC}"
echo ""

# ==================== 执行命令 ====================
if [[ "$DRY_RUN" == "true" ]]; then
    log_info "🏃 模拟运行模式 - 仅显示命令:"
    echo ""
    echo -e "${YELLOW}$PYTHON_CMD${NC}"
    echo ""
else
    log_info "🚀 开始性能分析..."
    echo ""
    
    # 记录开始时间
    START_TIME=$(date +%s)
    
    # 执行命令
    if eval "$PYTHON_CMD"; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        echo ""
        log_info "✅ 分析完成！总耗时: ${DURATION}秒"
        
        if [[ -n "$CSV_OUTPUT" && -f "$CSV_OUTPUT" ]]; then
            log_info "📊 CSV结果已保存: $CSV_OUTPUT"
        fi
        
        if [[ -f "profile_pipeline.log" ]]; then
            log_info "📝 详细日志已保存: profile_pipeline.log"
        fi
    else
        log_error "❌ 分析失败！"
        exit 1
    fi
fi

echo ""
log_info "🎯 脚本执行完成"