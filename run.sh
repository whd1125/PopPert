#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR"

declare -A DS_MODE DS_PERT_COL DS_CELL_LINE DS_CELL_LINE_COL DS_CTRL_VAL DS_H5AD

DS_MODE[replogle]="gene"
DS_PERT_COL[replogle]="gene"
DS_CELL_LINE[replogle]="hepg2"
DS_CELL_LINE_COL[replogle]="cell_line"
DS_CTRL_VAL[replogle]="non-targeting"
DS_H5AD[replogle]="replogle.h5ad"

DS_MODE[adamson]="gene"
DS_PERT_COL[adamson]="gene"
DS_CELL_LINE[adamson]="K562"
DS_CELL_LINE_COL[adamson]="cell_line"
DS_CTRL_VAL[adamson]="ctrl"
DS_H5AD[adamson]="adamson.h5ad"

DS_MODE[norman]="gene"
DS_PERT_COL[norman]="gene"
DS_CELL_LINE[norman]="K562"
DS_CELL_LINE_COL[norman]="cell_line"
DS_CTRL_VAL[norman]="ctrl"
DS_H5AD[norman]="norman.h5ad"

DS_MODE[sciplex]="drug"
DS_PERT_COL[sciplex]="condition"
DS_CELL_LINE[sciplex]="MCF7"
DS_CELL_LINE_COL[sciplex]="cell_type"
DS_CTRL_VAL[sciplex]="control"
DS_H5AD[sciplex]="sciplex.h5ad"

RUN_TAG="${RUN_TAG:-copula}"
DEVICE="${DEVICE:-cuda:0}"
N_COMPONENTS="${N_COMPONENTS:-2}"
N_HVG=5000
BATCH_SIZE=16
EPOCHS=100
D_MODEL=256
NHEAD=4
NLAYERS=4
DIM_FF=$((4 * D_MODEL))
N_SAMPLES_EVAL=512
N_SAMPLES_LOSS="${N_SAMPLES_LOSS:-256}"
COPULA_RANK="${COPULA_RANK:-32}"
COPULA_LAMBDA="${COPULA_LAMBDA:-0.2}"

SKIP_PREPROCESS=0
SKIP_TRAIN=0
SKIP_INFERENCE=0
SKIP_EVAL=0
NO_COPULA=0
NO_TRUNCATED=0
MARGINAL="zigmm"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)            DEVICE="$2"; shift 2 ;;
        --copula_lambda)     COPULA_LAMBDA="$2"; shift 2 ;;
        --copula_rank)       COPULA_RANK="$2"; shift 2 ;;
        --n_components)      N_COMPONENTS="$2"; shift 2 ;;
        --batch_size)        BATCH_SIZE="$2"; shift 2 ;;
        --epochs)            EPOCHS="$2"; shift 2 ;;
        --d_model)           D_MODEL="$2"; DIM_FF=$((4 * D_MODEL)); shift 2 ;;
        --nlayers)           NLAYERS="$2"; shift 2 ;;
        --nhead)             NHEAD="$2"; shift 2 ;;
        --run_tag)           RUN_TAG="$2"; shift 2 ;;
        --skip_preprocess)   SKIP_PREPROCESS=1; shift ;;
        --skip_train)        SKIP_TRAIN=1; shift ;;
        --skip_inference)    SKIP_INFERENCE=1; shift ;;
        --skip_eval)         SKIP_EVAL=1; shift ;;
        --no_copula)         NO_COPULA=1; shift ;;
        --no_truncated)      NO_TRUNCATED=1; shift ;;
        --marginal)          MARGINAL="$2"; shift 2 ;;
        --help)              echo "Usage: bash run.sh DATASET [flags]"; echo "Datasets: replogle adamson norman sciplex"; echo "Options: --marginal zigmm|zinb"; exit 0 ;;
        -*)                  echo "Unknown flag: $1"; exit 1 ;;
        *)                   DATASET_NAME="$1"; shift ;;
    esac
done

DATASET_NAME="${DATASET_NAME:-replogle}"

if [ -z "${DS_MODE[$DATASET_NAME]:-}" ]; then
    echo "Error: unknown dataset '$DATASET_NAME'.  Supported: replogle, adamson, norman, sciplex"
    exit 1
fi

MODE="${DS_MODE[$DATASET_NAME]}"
PERT_COL="${DS_PERT_COL[$DATASET_NAME]}"
CELL_LINE="${DS_CELL_LINE[$DATASET_NAME]}"
CELL_LINE_COL="${DS_CELL_LINE_COL[$DATASET_NAME]}"
CTRL_VAL="${DS_CTRL_VAL[$DATASET_NAME]}"
H5AD_NAME="${DS_H5AD[$DATASET_NAME]}"

if [ "$MARGINAL" = "zinb" ]; then
    PREPROCESS_SCRIPT="zinb/preprocess.py"
    TRAIN_SCRIPT="zinb/train.py"
    INFERENCE_SCRIPT="zinb/inference.py"
    PROCESSED_SUFFIX="_zinb"
    N_COMP_ARG=""                      # ZINB has no n_components
    CELL_LINE_COL_ARG=""               # ZINB inference has no cell_line_col
    PERT_COL_ARG=""                    # ZINB inference has no perturbation_col
    if [ "$MODE" = "drug" ]; then
        echo "Error: ZINB marginal does not support drug mode."
        exit 1
    fi
else
    PREPROCESS_SCRIPT="preprocess.py"
    TRAIN_SCRIPT="train.py"
    INFERENCE_SCRIPT="inference.py"
    PROCESSED_SUFFIX=""
    N_COMP_ARG="--n_components $N_COMPONENTS"
    CELL_LINE_COL_ARG="--cell_line_col $CELL_LINE_COL"
    PERT_COL_ARG="--perturbation_col $PERT_COL"
fi

PYTHON="${PYTHON:-python3}"

DATA_DIR="$SCRIPT_DIR/data"

if [ "$MARGINAL" = "zinb" ]; then
    RAW_H5AD="$DATA_DIR/$DATASET_NAME/${DATASET_NAME}_raw.h5ad"
    if [ -f "$RAW_H5AD" ]; then
        H5AD_PATH="$RAW_H5AD"
    else
        H5AD_PATH="$DATA_DIR/$DATASET_NAME/$H5AD_NAME"
        echo "  WARNING: $_raw.h5ad not found, falling back to $H5AD_NAME"
    fi
else
    H5AD_PATH="$DATA_DIR/$DATASET_NAME/$H5AD_NAME"
fi
TOML_PATH="$SCRIPT_DIR/configs/$DATASET_NAME.toml"
VOCAB_PATH="$SCRIPT_DIR/vocab.json"
PROCESSED_DATA="$DATA_DIR/processed/processed_${DATASET_NAME}${PROCESSED_SUFFIX}.pt"
RUN_DIR="$DATA_DIR/../results/$DATASET_NAME/ours_${RUN_TAG}_lam${COPULA_LAMBDA}"
CKPT_DIR="$RUN_DIR/checkpoints"
MODEL_PATH="$CKPT_DIR/best_model.pt"

mkdir -p "$DATA_DIR/processed" "$CKPT_DIR"

echo "============================================================"
echo "  PopPert Pipeline"
echo "  Dataset:   $DATASET_NAME  (mode: $MODE)"
echo "  Model:     d=$D_MODEL, layers=$NLAYERS, heads=$NHEAD, K=$N_COMPONENTS"
echo "  Copula:    rank=$COPULA_RANK, λ=$COPULA_LAMBDA"$( [[ $NO_COPULA -eq 1 ]] && echo " (DISABLED)" )
echo "  Device:    $DEVICE"
echo "  Output:    $RUN_DIR"
echo "============================================================"
echo ""

if [ "$SKIP_PREPROCESS" -eq 1 ]; then
    echo "[1/4] Preprocessing SKIPPED (--skip_preprocess)"
else
    echo "[1/4] Preprocessing..."

    NEED_REPROCESS=0
    if [ -f "$PROCESSED_DATA" ]; then
        echo "  Processed data already exists: $PROCESSED_DATA"
        echo "  (delete it or use --skip_preprocess to force re-run)"
    else
        NEED_REPROCESS=1
    fi

    if [ "$NEED_REPROCESS" -eq 1 ]; then
        PREPROCESS_EXTRA=""
        if [ "$MODE" = "drug" ]; then
            PREPROCESS_EXTRA="--perturb_mode drug"
        fi
        if [ "$NO_TRUNCATED" -eq 1 ]; then
            PREPROCESS_EXTRA="$PREPROCESS_EXTRA --no-truncated"
        fi
        if [ "$NO_COPULA" -eq 1 ]; then
            PREPROCESS_EXTRA="$PREPROCESS_EXTRA --copula_rank 0"
        fi

        $PYTHON $PREPROCESS_SCRIPT \
            --dataset "$DATASET_NAME" \
            --h5ad_path "$H5AD_PATH" \
            --toml_path "$TOML_PATH" \
            --output_path "$PROCESSED_DATA" \
            --vocab_path "$VOCAB_PATH" \
            $N_COMP_ARG \
            --n_hvg "$N_HVG" \
            --control_value "$CTRL_VAL" \
            --perturbation_col "$PERT_COL" \
            --cell_line_col "$CELL_LINE_COL" \
            --copula_rank "$COPULA_RANK" \
            --n_jobs 10 \
            $PREPROCESS_EXTRA

        if [ $? -ne 0 ]; then echo "Preprocessing failed!"; exit 1; fi
        echo "  Preprocessing complete: $PROCESSED_DATA"
    fi
fi

if [ "$SKIP_TRAIN" -eq 1 ]; then
    echo "[2/4] Training SKIPPED (--skip_train)"
else
    echo "[2/4] Training..."

    TRAIN_EXTRA=""
    if [ "$MODE" = "drug" ]; then
        TRAIN_EXTRA="--perturb_mode drug"
    fi
    if [ "$NO_COPULA" -eq 1 ]; then
        TRAIN_EXTRA="$TRAIN_EXTRA --no_copula"
    fi

    $PYTHON $TRAIN_SCRIPT \
        --dataset "$DATASET_NAME" \
        --data_path "$PROCESSED_DATA" \
        --vocab_path "$VOCAB_PATH" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --d_model "$D_MODEL" \
        --nhead "$NHEAD" \
        --nlayers "$NLAYERS" \
        --dim_feedforward "$DIM_FF" \
        $N_COMP_ARG \
        --copula_lambda "$COPULA_LAMBDA" \
        --n_samples_loss "$N_SAMPLES_LOSS" \
        --save_dir "$CKPT_DIR" \
        --device "$DEVICE" \
        $TRAIN_EXTRA

    if [ $? -ne 0 ]; then echo "Training failed!"; exit 1; fi
    echo "  Training complete."
fi

if [ "$SKIP_INFERENCE" -eq 1 ]; then
    echo "[3/4] Inference SKIPPED (--skip_inference)"
else
    echo "[3/4] Inference..."

    INF_EXTRA=""
    if [ "$MODE" = "drug" ]; then
        INF_EXTRA="--perturb_mode drug"
    fi

    $PYTHON $INFERENCE_SCRIPT \
        --dataset "$DATASET_NAME" \
        --data_path "$PROCESSED_DATA" \
        --model_path "$MODEL_PATH" \
        --vocab_path "$VOCAB_PATH" \
        --save_dir "$RUN_DIR/inference" \
        --raw_data_path "$H5AD_PATH" \
        --n_samples "$N_SAMPLES_EVAL" \
        --cell_line "$CELL_LINE" \
        $CELL_LINE_COL_ARG \
        --control_value "$CTRL_VAL" \
        $PERT_COL_ARG \
        --d_model "$D_MODEL" \
        --nhead "$NHEAD" \
        --nlayers "$NLAYERS" \
        $INF_EXTRA

    if [ $? -ne 0 ]; then echo "Inference failed!"; exit 1; fi
    echo "  Inference complete."
fi

if [ "$SKIP_EVAL" -eq 1 ]; then
    echo "[4/4] Evaluation SKIPPED (--skip_eval)"
else
    echo "[4/4] Evaluation (cell-eval)..."

    PRED_PATH="$RUN_DIR/inference/inference_pred.h5ad"
    REAL_PATH="$RUN_DIR/inference/inference_real.h5ad"

    if [ ! -f "$PRED_PATH" ] || [ ! -f "$REAL_PATH" ]; then
        echo "  Inference outputs not found; skipping evaluation."
    elif command -v cell-eval &> /dev/null; then
        cell-eval run \
            -ap "$PRED_PATH" \
            -ar "$REAL_PATH" \
            --control-pert "$CTRL_VAL" \
            -o "$RUN_DIR/cell_eval"
        echo "  cell-eval results: $RUN_DIR/cell_eval"
    else
        echo "  cell-eval not found. Install: pip install cell-eval"
        echo "  Then run manually:"
        echo "    cell-eval run -ap $PRED_PATH -ar $REAL_PATH --control-pert $CTRL_VAL -o $RUN_DIR/cell_eval"
    fi
fi

echo ""
echo "============================================================"
echo "  Pipeline complete."
echo "  Results: $RUN_DIR"
echo "============================================================"
