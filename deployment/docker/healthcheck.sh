#!/bin/bash
# Health check script for SDLC automation container

set -e

# Configuration
HEALTH_CHECK_URL="http://localhost:8000/health"
TIMEOUT=10
LOG_FILE="/app/logs/healthcheck.log"

# Logging function
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - HEALTHCHECK - $1" | tee -a "$LOG_FILE" 2>/dev/null || echo "$(date '+%Y-%m-%d %H:%M:%S') - HEALTHCHECK - $1"
}

# Check if SDLC processes are running
check_processes() {
    log_message "Checking SDLC processes..."
    
    # Check for Python SDLC processes
    if ! pgrep -f "sdlc_automation.py" > /dev/null; then
        log_message "ERROR: SDLC automation process not running"
        return 1
    fi
    
    log_message "SDLC processes check: PASSED"
    return 0
}

# Check system resources
check_resources() {
    log_message "Checking system resources..."
    
    # Check memory usage
    memory_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    memory_threshold=90.0
    
    if (( $(echo "$memory_usage > $memory_threshold" | bc -l) )); then
        log_message "WARNING: High memory usage: ${memory_usage}%"
    else
        log_message "Memory usage: ${memory_usage}% - OK"
    fi
    
    # Check disk space
    disk_usage=$(df /app | tail -1 | awk '{print $5}' | sed 's/%//')
    disk_threshold=85
    
    if [ "$disk_usage" -gt "$disk_threshold" ]; then
        log_message "WARNING: High disk usage: ${disk_usage}%"
    else
        log_message "Disk usage: ${disk_usage}% - OK"
    fi
    
    log_message "System resources check: PASSED"
    return 0
}

# Check file system permissions and directories
check_filesystem() {
    log_message "Checking filesystem..."
    
    # Check required directories exist and are writable
    required_dirs=("/app/data" "/app/logs" "/app/reports" "/app/tmp")
    
    for dir in "${required_dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            log_message "ERROR: Required directory missing: $dir"
            return 1
        fi
        
        if [ ! -w "$dir" ]; then
            log_message "ERROR: Directory not writable: $dir"
            return 1
        fi
    done
    
    log_message "Filesystem check: PASSED"
    return 0
}

# Check Python environment
check_python_environment() {
    log_message "Checking Python environment..."
    
    # Check if required packages are importable
    required_packages=("continual_transformer.sdlc.core" "continual_transformer.sdlc.automation")
    
    for package in "${required_packages[@]}"; do
        if ! python -c "import $package" 2>/dev/null; then
            log_message "ERROR: Cannot import required package: $package"
            return 1
        fi
    done
    
    log_message "Python environment check: PASSED"
    return 0
}

# Check database connectivity (SQLite)
check_database() {
    log_message "Checking database connectivity..."
    
    # Test SQLite database creation and basic operations
    test_db="/app/tmp/healthcheck_test.db"
    
    if ! sqlite3 "$test_db" "CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY); INSERT INTO test DEFAULT VALUES; SELECT COUNT(*) FROM test;" > /dev/null 2>&1; then
        log_message "ERROR: Database connectivity test failed"
        return 1
    fi
    
    # Cleanup test database
    rm -f "$test_db" 2>/dev/null || true
    
    log_message "Database connectivity check: PASSED"
    return 0
}

# Main health check function
main() {
    log_message "Starting health check..."
    
    # Run all health checks
    health_checks=(
        "check_processes"
        "check_resources"
        "check_filesystem"
        "check_python_environment"
        "check_database"
    )
    
    for check in "${health_checks[@]}"; do
        if ! $check; then
            log_message "Health check FAILED: $check"
            exit 1
        fi
    done
    
    log_message "All health checks PASSED"
    exit 0
}

# Create log directory if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || true

# Run main health check
main "$@"