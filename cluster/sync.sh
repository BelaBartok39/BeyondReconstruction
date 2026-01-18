#!/bin/bash
# Sync project to/from cluster
# Usage:
#   ./sync.sh push    - Push local changes to cluster
#   ./sync.sh pull    - Pull results from cluster
#   ./sync.sh status  - Check sync status

set -e

# Configuration
CLUSTER_HOST="bigblue.memphis.edu"
CLUSTER_USER="ndrdmond"
SSH_KEY="$HOME/.ssh/school_gpu_key"
LOCAL_DIR="$(dirname "$(dirname "$(realpath "$0")")")"
REMOTE_DIR="~/CLP_Project"

# SSH/rsync options
SSH_OPTS="-i $SSH_KEY"
RSYNC_OPTS="-avz --progress -e \"ssh $SSH_OPTS\""

# Exclusions for push (don't overwrite cluster data)
PUSH_EXCLUDE="--exclude='.git' --exclude='*.pyc' --exclude='__pycache__' \
              --exclude='.pytest_cache' --exclude='*.egg-info' \
              --exclude='checkpoints/' --exclude='logs/' --exclude='results/' \
              --exclude='.venv' --exclude='venv' --exclude='*.pt' \
              --exclude='wandb/' --exclude='.ipynb_checkpoints'"

# Exclusions for pull (only get results)
PULL_INCLUDE="--include='checkpoints/***' --include='logs/***' --include='results/***' \
              --include='*.json' --include='*.png' --exclude='*'"

push_to_cluster() {
    echo "Pushing code to cluster..."
    echo "Local: $LOCAL_DIR"
    echo "Remote: $CLUSTER_USER@$CLUSTER_HOST:$REMOTE_DIR"
    echo ""

    eval rsync $RSYNC_OPTS $PUSH_EXCLUDE "$LOCAL_DIR/" "$CLUSTER_USER@$CLUSTER_HOST:$REMOTE_DIR/"

    echo ""
    echo "Push complete!"
}

pull_from_cluster() {
    echo "Pulling results from cluster..."
    echo "Remote: $CLUSTER_USER@$CLUSTER_HOST:$REMOTE_DIR"
    echo "Local: $LOCAL_DIR"
    echo ""

    # Pull checkpoints
    eval rsync $RSYNC_OPTS "$CLUSTER_USER@$CLUSTER_HOST:$REMOTE_DIR/checkpoints/" "$LOCAL_DIR/checkpoints/" 2>/dev/null || true

    # Pull logs
    eval rsync $RSYNC_OPTS "$CLUSTER_USER@$CLUSTER_HOST:$REMOTE_DIR/logs/" "$LOCAL_DIR/logs/" 2>/dev/null || true

    # Pull results
    eval rsync $RSYNC_OPTS "$CLUSTER_USER@$CLUSTER_HOST:$REMOTE_DIR/results/" "$LOCAL_DIR/results/" 2>/dev/null || true

    echo ""
    echo "Pull complete!"
}

check_status() {
    echo "Checking cluster connection..."
    ssh $SSH_OPTS "$CLUSTER_USER@$CLUSTER_HOST" "echo 'Connection successful!'; \
        echo ''; \
        echo 'Disk usage:'; \
        du -sh ~/CLP_Project 2>/dev/null || echo 'Project not found'; \
        echo ''; \
        echo 'Running jobs:'; \
        squeue -u $CLUSTER_USER 2>/dev/null || echo 'No jobs or squeue unavailable'"
}

show_help() {
    echo "RF Anomaly Detection - Cluster Sync Script"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  push    - Push local code to cluster"
    echo "  pull    - Pull results/checkpoints from cluster"
    echo "  status  - Check cluster connection and job status"
    echo "  ssh     - Open SSH session to cluster"
    echo ""
    echo "Configuration:"
    echo "  Cluster: $CLUSTER_USER@$CLUSTER_HOST"
    echo "  SSH Key: $SSH_KEY"
    echo "  Local:   $LOCAL_DIR"
    echo "  Remote:  $REMOTE_DIR"
}

case "$1" in
    push)
        push_to_cluster
        ;;
    pull)
        pull_from_cluster
        ;;
    status)
        check_status
        ;;
    ssh)
        echo "Connecting to cluster..."
        ssh $SSH_OPTS "$CLUSTER_USER@$CLUSTER_HOST"
        ;;
    *)
        show_help
        ;;
esac
