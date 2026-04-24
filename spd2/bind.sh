#!/bin/bash
set -euo pipefail

################################################################################
# Logging & Timing Setup
################################################################################
start_time=$(date +%s)
timestamp() {
    # Print a human-readable timestamp plus elapsed seconds since script start
    local now_s
    now_s=$(date +%s)
    local elapsed=$(( now_s - start_time ))
    echo "[bind.sh][$(date +'%Y-%m-%d %H:%M:%S')][+${elapsed}s]"
}

echo "$(timestamp) Starting bind.sh with args: $*"

################################################################################
# Usage
################################################################################
print_usage() {
    cat << EOF
${0} [options] [--] COMMAND [ARG...]

Control binding policy for each task. Assumes one rank will be launched for each GPU.

Options:
    --cpu=MODE
        * exclusive -- bind each rank to an exclusive set of cores near its GPU
        * exclusive,nosmt -- bind each rank to an exclusive set of cores near its GPU, without hyperthreading
        * node -- bind each rank to all cores in the NUMA node nearest its GPU [default]
        * *.sh -- bind each rank using the bash associative array bind_cpu_cores or bind_cpu_nodes from a file
        * off -- don't bind
    --mem=MODE
        * node -- bind each rank to the nearest NUMA node [default]
        * *.sh -- bind each rank using the bash associative array bind_mem from a file
        * off -- don't bind
    --ib=MODE
        * single -- bind each rank to a single IB device near its GPU
        * off -- don't bind [default]
    --cluster=CLUSTER
        Select which cluster is being used. May be required if system params cannot be detected.
EOF
}

################################################################################
# Argument parsing
################################################################################

cpu_mode='node'
mem_mode='node'
ib_mode='off'
cluster=''

echo "$(timestamp) Parsing arguments..."
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) print_usage; exit 0 ;;
        --cpu=*) cpu_mode="${1#*=}"; shift ;;
        --cpu)   cpu_mode="$2"; shift 2 ;;
        --mem=*) mem_mode="${1#*=}"; shift ;;
        --mem)   mem_mode="$2"; shift 2 ;;
        --ib=*)  ib_mode="${1#*=}"; shift ;;
        --ib)    ib_mode="$2"; shift 2 ;;
        --cluster=*) cluster="${1#*=}"; shift ;;
        --cluster)   cluster="$2"; shift 2 ;;
        --) shift; break ;;
        *) break ;;
    esac
done

if [ $# -lt 1 ]; then
    echo "$(timestamp) ERROR: no command given" >&2
    print_usage
    exit 1
fi
user_command=( "$@" )

echo "$(timestamp) Done parsing arguments."
echo "$(timestamp) CPU mode: $cpu_mode, MEM mode: $mem_mode, IB mode: $ib_mode, cluster: $cluster"
echo "$(timestamp) User command to exec: ${user_command[*]}"

################################################################################
# Get system params
################################################################################
echo "$(timestamp) Gathering system params..."

# Attempt to read local rank from various env variables
local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-}}}"
if [ -z "${local_rank}" ]; then
    echo "$(timestamp) ERROR: cannot read LOCAL_RANK (or SLURM_LOCALID/OMPI_COMM_WORLD_LOCAL_RANK) from env" >&2
    exit 1
fi

# Count GPUs on the node
num_gpus="$(nvidia-smi -i 0 --query-gpu=count --format=csv,noheader,nounits)"
if [ "${local_rank}" -ge "${num_gpus}" ]; then
    echo "$(timestamp) ERROR: local rank=${local_rank}, but only ${num_gpus} GPUs available" >&2
    exit 1
fi

# Gather CPU topology with lscpu
get_lscpu_value() {
    awk -F: "(\$1 == \"$1\"){gsub(/ /, \"\", \$2); print \$2; found=1} END{exit found!=1}"
}

lscpu_out="$(lscpu)"
num_sockets=$(get_lscpu_value "Socket(s)" <<< "${lscpu_out}")
num_nodes=$(get_lscpu_value "NUMA node(s)" <<< "${lscpu_out}")
cores_per_socket=$(get_lscpu_value "Core(s) per socket" <<< "${lscpu_out}")

echo "$(timestamp) CPU topology: num_sockets=${num_sockets}, num_nodes=${num_nodes}, cores_per_socket=${cores_per_socket}"

cores_per_node=$(( (num_sockets * cores_per_socket) / num_nodes ))
if [ "${num_gpus}" -gt 1 ]; then
    gpus_per_node=$(( num_gpus / num_nodes ))
else
    gpus_per_node=1
fi
cores_per_gpu=$(( cores_per_node / gpus_per_node ))
local_node=$(( local_rank / gpus_per_node ))

echo "$(timestamp) local_rank=${local_rank}, num_gpus=${num_gpus}, gpus_per_node=${gpus_per_node}"
echo "$(timestamp) cores_per_node=${cores_per_node}, cores_per_gpu=${cores_per_gpu}, local_node=${local_node}"

# IB device detection
declare -a ibdevs=()
case "${cluster}" in
    circe)
        ibdevs=(mlx5_1 mlx5_2 mlx5_3 mlx5_4 mlx5_7 mlx5_8 mlx5_9 mlx5_10)
        ;;
    selene)
        ibdevs=(mlx5_0 mlx5_1 mlx5_2 mlx5_3 mlx5_6 mlx5_7 mlx5_8 mlx5_9)
        ;;
    '')
        if ibstat_out="$(ibstat -l 2>/dev/null | sort -V)"; then
            mapfile -t ibdevs <<< "${ibstat_out}"
        fi
        ;;
    *)
        echo "$(timestamp) ERROR: Unknown cluster '${cluster}'" >&2
        exit 1
        ;;
esac
num_ibdevs="${#ibdevs[@]}"
echo "$(timestamp) IB devices detected: ${ibdevs[*]}"

################################################################################
# Build numactl args
################################################################################
echo "$(timestamp) Building numactl arguments..."
numactl_args=()

case "${cpu_mode}" in
    exclusive)
        numactl_args+=( "$(printf -- "--physcpubind=%u-%u,%u-%u" \
            $(( local_rank * cores_per_gpu )) \
            $(( (local_rank + 1) * cores_per_gpu - 1 )) \
            $(( local_rank * cores_per_gpu + (cores_per_gpu * gpus_per_node * num_nodes) )) \
            $(( (local_rank + 1) * cores_per_gpu + (cores_per_gpu * gpus_per_node * num_nodes) - 1 )) \
        )" )
        ;;
    exclusive,nosmt)
        numactl_args+=( "$(printf -- "--physcpubind=%u-%u" \
            $(( local_rank * cores_per_gpu )) \
            $(( (local_rank + 1) * cores_per_gpu - 1 )) \
        )" )
        ;;
    node)
        numactl_args+=( "--cpunodebind=${local_node}" )
        ;;
    *.sh)
        # Source a custom shell script that sets bind_cpu_cores[] or bind_cpu_nodes[]
        source "${cpu_mode}"
        if [ -n "${bind_cpu_cores:-}" ]; then
            numactl_args+=( "--physcpubind=${bind_cpu_cores[${local_rank}]}" )
        elif [ -n "${bind_cpu_nodes:-}" ]; then
            numactl_args+=( "--cpunodebind=${bind_cpu_nodes[${local_rank}]}" )
        else
            echo "$(timestamp) ERROR: invalid CPU affinity file ${cpu_mode}." >&2
            exit 1
        fi
        ;;
    off|'')
        ;;
    *)
        echo "$(timestamp) ERROR: invalid cpu mode '${cpu_mode}'" >&2
        print_usage
        exit 1
        ;;
esac

case "${mem_mode}" in
    node)
        numactl_args+=( "--membind=${local_node}" )
        ;;
    *.sh)
        source "${mem_mode}"
        if [ -z "${bind_mem:-}" ]; then
            echo "$(timestamp) ERROR: invalid memory affinity file ${mem_mode}." >&2
            exit 1
        fi
        numactl_args+=( "--membind=${bind_mem[${local_rank}]}" )
        ;;
    off|'')
        ;;
    *)
        echo "$(timestamp) ERROR: invalid mem mode '${mem_mode}'" >&2
        print_usage
        exit 1
        ;;
esac

case "${ib_mode}" in
    single)
        if [ "${num_ibdevs}" -eq 0 ]; then
            echo "$(timestamp) WARNING: --ib=single but 0 IB devices found; skipping IB binding." >&2
        else
            ibdev="${ibdevs[$(( local_rank * num_ibdevs / num_gpus ))]}"
            export OMPI_MCA_btl_openib_if_include="${OMPI_MCA_btl_openib_if_include:-$ibdev}"
            export UCX_NET_DEVICES="${UCX_NET_DEVICES:-$ibdev:1}"
            echo "$(timestamp) Setting IB device to '${ibdev}' for local_rank=${local_rank}"
        fi
        ;;
    off|'')
        ;;
    *)
        echo "$(timestamp) ERROR: invalid ib mode '${ib_mode}'" >&2
        print_usage
        exit 1
        ;;
esac

echo "$(timestamp) numactl_args = ${numactl_args[*]}"

################################################################################
# Exec final command
################################################################################
if [ "${#numactl_args[@]}" -gt 0 ]; then
    echo "$(timestamp) Running numactl with above args + user command..."
    set -x
    exec numactl "${numactl_args[@]}" -- "${user_command[@]}"
else
    echo "$(timestamp) No binding requested. Running command directly..."
    set -x
    exec "${user_command[@]}"
fi
