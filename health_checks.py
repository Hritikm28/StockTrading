from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import shutil
import psutil
from logging_config import get_logger


class HealthChecker:
    """System health monitoring for trading system"""
    
    def __init__(self):
        self.logger = get_logger()
        self.checks_passed = {}
    
    def check_data_freshness(self, data_dir='data/stocks', max_age_hours=24):

        try:
            data_path = Path(data_dir)
            
            # Check if directory exists
            if not data_path.exists():
                self.logger.log_error(
                    'DataFreshnessError',
                    'Data directory does not exist',
                    {'path': str(data_path)}
                )
                return False
            
            # Get all data files
            files = list(data_path.glob('*.parquet')) + list(data_path.glob('*.csv'))
            
            if not files:
                self.logger.log_error(
                    'DataFreshnessError',
                    'No data files found',
                    {'path': str(data_path)}
                )
                return False
            
            # Get most recent file
            most_recent = max(files, key=lambda f: f.stat().st_mtime)
            file_modified = datetime.fromtimestamp(most_recent.stat().st_mtime)
            file_age = datetime.now() - file_modified
            
            # Check if data is too old
            if file_age > timedelta(hours=max_age_hours):
                self.logger.logger.warning(
                    f"⚠️  Data is {file_age.days} days old - may be stale"
                )
                self.logger.logger.warning(
                    f"   Last updated: {file_modified.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                return False
            
            self.logger.logger.info(
                f"✅ Data freshness check passed "
                f"(last updated: {file_modified.strftime('%Y-%m-%d %H:%M')})"
            )
            return True
            
        except Exception as e:
            self.logger.log_error(
                'DataFreshnessCheckError',
                str(e),
                {'data_dir': data_dir}
            )
            return False
    
    def check_model_cache(self, cache_dir='model_cache', min_models=0):

        try:
            cache_path = Path(cache_dir)
            
            if not cache_path.exists():
                self.logger.logger.warning(
                    f"⚠️  Model cache directory not found: {cache_dir}"
                )
                self.logger.logger.info(
                    f"   This is OK for first run - models will be trained"
                )
                return True  # Not critical for first run
            
            # Count cached models
            model_files = list(cache_path.glob('*.pkl')) + list(cache_path.glob('*.joblib'))
            
            if len(model_files) < min_models:
                self.logger.logger.warning(
                    f"⚠️  Only {len(model_files)} models cached (expected >={min_models})"
                )
                return False
            
            self.logger.logger.info(
                f"✅ Model cache check passed ({len(model_files)} models cached)"
            )
            return True
            
        except Exception as e:
            self.logger.log_error(
                'ModelCacheCheckError',
                str(e),
                {'cache_dir': cache_dir}
            )
            return False
    
    def check_disk_space(self, min_gb=10, path='/'):

        try:
            total, used, free = shutil.disk_usage(path)
            free_gb = free // (2**30)  # Convert bytes to GB
            
            if free_gb < min_gb:
                self.logger.log_error(
                    'DiskSpaceError',
                    f'Low disk space: {free_gb}GB < {min_gb}GB required',
                    {'free_gb': free_gb, 'min_gb': min_gb, 'path': path}
                )
                return False
            
            self.logger.logger.info(
                f"✅ Disk space check passed ({free_gb}GB available)"
            )
            return True
            
        except Exception as e:
            self.logger.log_error(
                'DiskSpaceCheckError',
                str(e),
                {'path': path}
            )
            return False
    
    def check_memory_usage(self, max_percent=85):

        try:
            memory = psutil.virtual_memory()
            
            if memory.percent > max_percent:
                self.logger.logger.warning(
                    f"⚠️  High memory usage: {memory.percent:.1f}% > {max_percent}%"
                )
                self.logger.logger.warning(
                    f"   Available: {memory.available // (2**30)}GB "
                    f"/ Total: {memory.total // (2**30)}GB"
                )
                return False
            
            self.logger.logger.info(
                f"✅ Memory check passed ({memory.percent:.1f}% used, "
                f"{memory.available // (2**30)}GB available)"
            )
            return True
            
        except Exception as e:
            self.logger.log_error(
                'MemoryCheckError',
                str(e)
            )
            return False
    
    def check_cpu_usage(self, max_percent=90, interval=1):

        try:
            cpu_percent = psutil.cpu_percent(interval=interval)
            
            if cpu_percent > max_percent:
                self.logger.logger.warning(
                    f"⚠️  High CPU usage: {cpu_percent:.1f}% > {max_percent}%"
                )
                return False
            
            self.logger.logger.info(
                f"✅ CPU check passed ({cpu_percent:.1f}% used)"
            )
            return True
            
        except Exception as e:
            self.logger.log_error(
                'CPUCheckError',
                str(e)
            )
            return False
    
    def check_network_connectivity(self, test_hosts=None):

        if test_hosts is None:
            # Skip network check if no hosts specified
            return True
        
        try:
            import socket
            
            for host in test_hosts:
                try:
                    socket.create_connection((host, 80), timeout=3)
                except (socket.timeout, socket.error):
                    self.logger.logger.warning(
                        f"⚠️  Cannot reach {host}"
                    )
                    return False
            
            self.logger.logger.info(
                f"✅ Network check passed (all hosts reachable)"
            )
            return True
            
        except Exception as e:
            self.logger.log_error(
                'NetworkCheckError',
                str(e)
            )
            return False
    
    def run_all_checks(self, config=None):

        config = config or {}
        
        self.logger.logger.info("🏥 Running system health checks...")
        
        # Run all checks
        checks = {
            'Data Freshness': self.check_data_freshness(
                data_dir=config.get('data_dir', 'data/stocks'),
                max_age_hours=config.get('max_data_age_hours', 24)
            ),
            'Model Cache': self.check_model_cache(
                cache_dir=config.get('cache_dir', 'model_cache'),
                min_models=config.get('min_models', 0)
            ),
            'Disk Space': self.check_disk_space(
                min_gb=config.get('min_disk_gb', 10)
            ),
            'Memory Usage': self.check_memory_usage(
                max_percent=config.get('max_memory_percent', 85)
            ),
            'CPU Usage': self.check_cpu_usage(
                max_percent=config.get('max_cpu_percent', 90)
            ),
        }
        
        # Store results
        self.checks_passed = checks
        
        # Determine overall status
        all_passed = all(checks.values())
        
        if all_passed:
            self.logger.logger.info("✅ All health checks passed")
        else:
            failed = [k for k, v in checks.items() if not v]
            self.logger.logger.error(
                f"❌ Health checks failed: {', '.join(failed)}"
            )
            self.logger.logger.error(
                f"   Consider resolving issues before running analysis"
            )
        
        return all_passed
    
    def get_system_info(self):

        try:
            import platform
            
            info = {
                'timestamp': datetime.now().isoformat(),
                'platform': platform.system(),
                'platform_release': platform.release(),
                'platform_version': platform.version(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory': {
                    'total_gb': psutil.virtual_memory().total // (2**30),
                    'available_gb': psutil.virtual_memory().available // (2**30),
                    'percent': psutil.virtual_memory().percent
                },
                'disk': {
                    'total_gb': shutil.disk_usage('/').total // (2**30),
                    'free_gb': shutil.disk_usage('/').free // (2**30),
                    'percent': (shutil.disk_usage('/').used / shutil.disk_usage('/').total) * 100
                }
            }
            
            return info
            
        except Exception as e:
            self.logger.log_error('SystemInfoError', str(e))
            return {}


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == '__main__':
    # Example usage
    health = HealthChecker()
    
    # Run all checks with custom config
    config = {
        'data_dir': 'data/stocks',
        'max_data_age_hours': 48,  # Allow 2-day-old data
        'min_disk_gb': 5,
        'max_memory_percent': 90
    }
    
    if health.run_all_checks(config):
        print("\n✅ System is healthy - safe to proceed with analysis")
    else:
        print("\n❌ System health issues detected - review warnings above")
        print(f"\nFailed checks: {[k for k, v in health.checks_passed.items() if not v]}")
    
    # Print system info
    print("\n📊 System Information:")
    import json
    print(json.dumps(health.get_system_info(), indent=2))