"""Storage management for continual learning models and data."""

import torch
import json
import pickle
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging
from datetime import datetime
import sqlite3
import hashlib
from dataclasses import dataclass, asdict
import os

logger = logging.getLogger(__name__)


@dataclass
class ModelCheckpoint:
    """Model checkpoint metadata."""
    checkpoint_id: str
    task_id: str
    epoch: int
    step: int
    loss: float
    accuracy: float
    model_path: str
    config_path: str
    created_at: datetime
    file_size_mb: float
    model_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelCheckpoint':
        """Create from dictionary."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


class ModelCheckpointManager:
    """Manages model checkpoints for continual learning."""
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 10,
        save_best_only: bool = False
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        
        # Metadata database
        self.db_path = self.checkpoint_dir / "checkpoints.db"
        self._init_database()
        
        # Best checkpoint tracking
        self.best_checkpoint: Optional[ModelCheckpoint] = None
        self._load_best_checkpoint()
        
        logger.info(f"Initialized ModelCheckpointManager at {checkpoint_dir}")
    
    def _init_database(self):
        """Initialize SQLite database for checkpoint metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    epoch INTEGER NOT NULL,
                    step INTEGER NOT NULL,
                    loss REAL NOT NULL,
                    accuracy REAL NOT NULL,
                    model_path TEXT NOT NULL,
                    config_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    file_size_mb REAL NOT NULL,
                    model_hash TEXT NOT NULL
                )
            """)
            conn.commit()
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def save_checkpoint(
        self,
        model,
        task_id: str,
        epoch: int,
        step: int,
        loss: float,
        accuracy: float,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save model checkpoint."""
        
        # Generate checkpoint ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_id = f"{task_id}_{epoch:03d}_{step:06d}_{timestamp}"
        
        # Create checkpoint directory
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save model state
        model_path = checkpoint_path / "model.pt"
        config_path = checkpoint_path / "config.json"
        
        # Prepare checkpoint data
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'task_id': task_id,
            'epoch': epoch,
            'step': step,
            'loss': loss,
            'accuracy': accuracy,
            'created_at': datetime.now().isoformat()
        }
        
        if additional_data:
            checkpoint_data.update(additional_data)
        
        # Save model
        torch.save(checkpoint_data, model_path)
        
        # Save configuration
        config_data = {
            'checkpoint_id': checkpoint_id,
            'task_id': task_id,
            'epoch': epoch,
            'step': step,
            'loss': loss,
            'accuracy': accuracy,
            'model_config': model.config.to_dict() if hasattr(model, 'config') else {}
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Compute file size and hash
        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        model_hash = self._compute_file_hash(model_path)
        
        # Create checkpoint metadata
        checkpoint = ModelCheckpoint(
            checkpoint_id=checkpoint_id,
            task_id=task_id,
            epoch=epoch,
            step=step,
            loss=loss,
            accuracy=accuracy,
            model_path=str(model_path),
            config_path=str(config_path),
            created_at=datetime.now(),
            file_size_mb=file_size_mb,
            model_hash=model_hash
        )
        
        # Save to database
        self._save_checkpoint_metadata(checkpoint)
        
        # Update best checkpoint
        if self.best_checkpoint is None or accuracy > self.best_checkpoint.accuracy:
            self.best_checkpoint = checkpoint
            self._save_best_checkpoint_link(checkpoint)
        
        # Clean up old checkpoints
        if not self.save_best_only:
            self._cleanup_old_checkpoints(task_id)
        
        logger.info(f"Saved checkpoint {checkpoint_id} (accuracy: {accuracy:.4f})")
        return checkpoint_id
    
    def _save_checkpoint_metadata(self, checkpoint: ModelCheckpoint):
        """Save checkpoint metadata to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO checkpoints VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                checkpoint.checkpoint_id,
                checkpoint.task_id,
                checkpoint.epoch,
                checkpoint.step,
                checkpoint.loss,
                checkpoint.accuracy,
                checkpoint.model_path,
                checkpoint.config_path,
                checkpoint.created_at.isoformat(),
                checkpoint.file_size_mb,
                checkpoint.model_hash
            ))
            conn.commit()
    
    def _save_best_checkpoint_link(self, checkpoint: ModelCheckpoint):
        """Create symlink to best checkpoint."""
        best_dir = self.checkpoint_dir / "best"
        best_dir.mkdir(exist_ok=True)
        
        # Remove existing links
        for file in best_dir.glob("*"):
            if file.is_symlink():
                file.unlink()
        
        # Create new links
        model_src = Path(checkpoint.model_path)
        config_src = Path(checkpoint.config_path)
        
        (best_dir / "model.pt").symlink_to(model_src.resolve())
        (best_dir / "config.json").symlink_to(config_src.resolve())
        
        # Save best checkpoint info
        with open(best_dir / "best_info.json", 'w') as f:
            json.dump(checkpoint.to_dict(), f, indent=2)
    
    def _cleanup_old_checkpoints(self, task_id: str):
        """Remove old checkpoints beyond max limit."""
        checkpoints = self.list_checkpoints(task_id)
        
        if len(checkpoints) > self.max_checkpoints:
            # Sort by accuracy (ascending) to keep best ones
            checkpoints.sort(key=lambda x: x.accuracy)
            
            # Remove worst checkpoints
            to_remove = checkpoints[:-self.max_checkpoints]
            
            for checkpoint in to_remove:
                self.delete_checkpoint(checkpoint.checkpoint_id)
    
    def load_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """Load checkpoint data."""
        checkpoint = self.get_checkpoint(checkpoint_id)
        if checkpoint is None:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        return torch.load(checkpoint.model_path, map_location='cpu')
    
    def load_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load best checkpoint."""
        if self.best_checkpoint is None:
            return None
        
        return self.load_checkpoint(self.best_checkpoint.checkpoint_id)
    
    def get_checkpoint(self, checkpoint_id: str) -> Optional[ModelCheckpoint]:
        """Get checkpoint metadata."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM checkpoints WHERE checkpoint_id = ?",
                (checkpoint_id,)
            )
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            return ModelCheckpoint(
                checkpoint_id=row[0],
                task_id=row[1],
                epoch=row[2],
                step=row[3],
                loss=row[4],
                accuracy=row[5],
                model_path=row[6],
                config_path=row[7],
                created_at=datetime.fromisoformat(row[8]),
                file_size_mb=row[9],
                model_hash=row[10]
            )
    
    def list_checkpoints(self, task_id: Optional[str] = None) -> List[ModelCheckpoint]:
        """List all checkpoints, optionally filtered by task."""
        with sqlite3.connect(self.db_path) as conn:
            if task_id:
                cursor = conn.execute(
                    "SELECT * FROM checkpoints WHERE task_id = ? ORDER BY created_at DESC",
                    (task_id,)
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM checkpoints ORDER BY created_at DESC"
                )
            
            checkpoints = []
            for row in cursor.fetchall():
                checkpoints.append(ModelCheckpoint(
                    checkpoint_id=row[0],
                    task_id=row[1],
                    epoch=row[2],
                    step=row[3],
                    loss=row[4],
                    accuracy=row[5],
                    model_path=row[6],
                    config_path=row[7],
                    created_at=datetime.fromisoformat(row[8]),
                    file_size_mb=row[9],
                    model_hash=row[10]
                ))
            
            return checkpoints
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint."""
        checkpoint = self.get_checkpoint(checkpoint_id)
        if checkpoint is None:
            return False
        
        # Remove files
        checkpoint_dir = Path(checkpoint.model_path).parent
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        
        # Remove from database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM checkpoints WHERE checkpoint_id = ?",
                (checkpoint_id,)
            )
            conn.commit()
        
        logger.info(f"Deleted checkpoint {checkpoint_id}")
        return True
    
    def _load_best_checkpoint(self):
        """Load best checkpoint info on initialization."""
        best_info_path = self.checkpoint_dir / "best" / "best_info.json"
        if best_info_path.exists():
            try:
                with open(best_info_path, 'r') as f:
                    data = json.load(f)
                    self.best_checkpoint = ModelCheckpoint.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load best checkpoint info: {e}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        checkpoints = self.list_checkpoints()
        
        total_size_mb = sum(cp.file_size_mb for cp in checkpoints)
        
        stats = {
            'total_checkpoints': len(checkpoints),
            'total_size_mb': total_size_mb,
            'max_checkpoints': self.max_checkpoints,
            'save_best_only': self.save_best_only,
            'best_checkpoint': self.best_checkpoint.to_dict() if self.best_checkpoint else None,
            'checkpoints_by_task': {}
        }
        
        # Group by task
        for checkpoint in checkpoints:
            task_id = checkpoint.task_id
            if task_id not in stats['checkpoints_by_task']:
                stats['checkpoints_by_task'][task_id] = {
                    'count': 0,
                    'size_mb': 0,
                    'best_accuracy': 0
                }
            
            task_stats = stats['checkpoints_by_task'][task_id]
            task_stats['count'] += 1
            task_stats['size_mb'] += checkpoint.file_size_mb
            task_stats['best_accuracy'] = max(task_stats['best_accuracy'], checkpoint.accuracy)
        
        return stats


class TaskDataStorage:
    """Storage for task-specific data and metadata."""
    
    def __init__(self, storage_dir: Union[str, Path]):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Task metadata database
        self.db_path = self.storage_dir / "tasks.db"
        self._init_database()
        
        logger.info(f"Initialized TaskDataStorage at {storage_dir}")
    
    def _init_database(self):
        """Initialize task metadata database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    num_samples INTEGER NOT NULL,
                    num_classes INTEGER NOT NULL,
                    data_path TEXT NOT NULL,
                    metadata_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    data_hash TEXT NOT NULL
                )
            """)
            conn.commit()
    
    def save_task_data(
        self,
        task_id: str,
        data: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save task data and metadata."""
        
        # Create task directory
        task_dir = self.storage_dir / task_id
        task_dir.mkdir(exist_ok=True)
        
        # Save data
        data_path = task_dir / "data.json"
        with open(data_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Save metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'task_id': task_id,
            'num_samples': len(data),
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        })
        
        metadata_path = task_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Compute data hash
        data_hash = self._compute_data_hash(data)
        
        # Determine task properties
        task_type = metadata.get('task_type', 'classification')
        num_classes = len(set(item.get('label', 0) for item in data))
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task_id,
                task_type,
                len(data),
                num_classes,
                str(data_path),
                str(metadata_path),
                metadata['created_at'],
                metadata['updated_at'],
                data_hash
            ))
            conn.commit()
        
        logger.info(f"Saved task data '{task_id}' with {len(data)} samples")
        return str(task_dir)
    
    def load_task_data(self, task_id: str) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
        """Load task data and metadata."""
        task_info = self.get_task_info(task_id)
        if task_info is None:
            return None
        
        # Load data
        with open(task_info['data_path'], 'r') as f:
            data = json.load(f)
        
        # Load metadata
        with open(task_info['metadata_path'], 'r') as f:
            metadata = json.load(f)
        
        return data, metadata
    
    def get_task_info(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task information."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM tasks WHERE task_id = ?",
                (task_id,)
            )
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            return {
                'task_id': row[0],
                'task_type': row[1],
                'num_samples': row[2],
                'num_classes': row[3],
                'data_path': row[4],
                'metadata_path': row[5],
                'created_at': row[6],
                'updated_at': row[7],
                'data_hash': row[8]
            }
    
    def list_tasks(self) -> List[Dict[str, Any]]:
        """List all stored tasks."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM tasks ORDER BY created_at")
            
            tasks = []
            for row in cursor.fetchall():
                tasks.append({
                    'task_id': row[0],
                    'task_type': row[1],
                    'num_samples': row[2],
                    'num_classes': row[3],
                    'data_path': row[4],
                    'metadata_path': row[5],
                    'created_at': row[6],
                    'updated_at': row[7],
                    'data_hash': row[8]
                })
            
            return tasks
    
    def delete_task_data(self, task_id: str) -> bool:
        """Delete task data."""
        task_dir = self.storage_dir / task_id
        
        if task_dir.exists():
            shutil.rmtree(task_dir)
        
        # Remove from database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
            conn.commit()
        
        logger.info(f"Deleted task data '{task_id}'")
        return True
    
    def _compute_data_hash(self, data: List[Dict[str, Any]]) -> str:
        """Compute hash of data for integrity checking."""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()


class MetricsStorage:
    """Storage for training and evaluation metrics."""
    
    def __init__(self, storage_dir: Union[str, Path]):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics database
        self.db_path = self.storage_dir / "metrics.db"
        self._init_database()
        
        logger.info(f"Initialized MetricsStorage at {storage_dir}")
    
    def _init_database(self):
        """Initialize metrics database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    epoch INTEGER NOT NULL,
                    step INTEGER NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            
            # Create index for faster queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_run_task 
                ON metrics(run_id, task_id)
            """)
            
            conn.commit()
    
    def log_metrics(
        self,
        run_id: str,
        task_id: str,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        metric_type: str = "train"
    ):
        """Log metrics for a training run."""
        
        timestamp = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            for metric_name, metric_value in metrics.items():
                conn.execute("""
                    INSERT INTO metrics (run_id, task_id, epoch, step, metric_name, metric_value, metric_type, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (run_id, task_id, epoch, step, metric_name, metric_value, metric_type, timestamp))
            
            conn.commit()
    
    def get_metrics(
        self,
        run_id: str,
        task_id: Optional[str] = None,
        metric_name: Optional[str] = None,
        metric_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get metrics with optional filtering."""
        
        query = "SELECT * FROM metrics WHERE run_id = ?"
        params = [run_id]
        
        if task_id:
            query += " AND task_id = ?"
            params.append(task_id)
        
        if metric_name:
            query += " AND metric_name = ?"
            params.append(metric_name)
        
        if metric_type:
            query += " AND metric_type = ?"
            params.append(metric_type)
        
        query += " ORDER BY task_id, epoch, step"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            
            metrics = []
            for row in cursor.fetchall():
                metrics.append({
                    'id': row[0],
                    'run_id': row[1],
                    'task_id': row[2],
                    'epoch': row[3],
                    'step': row[4],
                    'metric_name': row[5],
                    'metric_value': row[6],
                    'metric_type': row[7],
                    'timestamp': row[8]
                })
            
            return metrics
    
    def get_run_summary(self, run_id: str) -> Dict[str, Any]:
        """Get summary statistics for a run."""
        metrics = self.get_metrics(run_id)
        
        if not metrics:
            return {}
        
        # Group by task and metric type
        summary = {
            'run_id': run_id,
            'total_metrics': len(metrics),
            'tasks': set(m['task_id'] for m in metrics),
            'metric_types': set(m['metric_type'] for m in metrics),
            'final_metrics': {}
        }
        
        # Get final metrics for each task
        for task_id in summary['tasks']:
            task_metrics = [m for m in metrics if m['task_id'] == task_id]
            
            # Get latest metrics for this task
            latest_epoch = max(m['epoch'] for m in task_metrics)
            latest_metrics = [m for m in task_metrics if m['epoch'] == latest_epoch]
            
            final_task_metrics = {}
            for metric in latest_metrics:
                final_task_metrics[metric['metric_name']] = metric['metric_value']
            
            summary['final_metrics'][task_id] = final_task_metrics
        
        return summary
    
    def export_metrics(self, run_id: str, output_path: Union[str, Path]):
        """Export metrics to JSON file."""
        metrics = self.get_metrics(run_id)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Exported {len(metrics)} metrics to {output_path}")
    
    def list_runs(self) -> List[str]:
        """List all run IDs."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT DISTINCT run_id FROM metrics ORDER BY run_id")
            return [row[0] for row in cursor.fetchall()]