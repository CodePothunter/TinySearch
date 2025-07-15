"""
Hot-update capabilities for TinySearch Flow Controller
"""
from typing import Dict, List, Any, Optional, Union, Callable, Set, TYPE_CHECKING
import os
import time
import threading
import logging
from pathlib import Path
import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent, FileMovedEvent

from tinysearch.validation import ValidationError

logger = logging.getLogger(__name__)


class FileChangeHandler(FileSystemEventHandler):
    """
    Handler for file system events (creation, modification, deletion)
    """
    
    def __init__(
        self,
        flow_controller,
        file_extensions: List[str],
        process_delay: float = 1.0,
        on_update_callback: Optional[Callable] = None
    ):
        """
        Initialize the file change handler
        
        Args:
            flow_controller: FlowController instance to process file changes
            file_extensions: List of file extensions to monitor
            process_delay: Delay in seconds before processing changes
            on_update_callback: Optional callback function to call after processing
        """
        self.flow_controller = flow_controller
        self.file_extensions = [ext.lower() for ext in file_extensions]
        self.process_delay = process_delay
        self.on_update_callback = on_update_callback
        
        # Track pending updates to avoid duplicate processing
        self.pending_updates = {}
        self.pending_deletions = set()
        self.lock = threading.Lock()
        
        # Create a timer thread for delayed processing
        self.timer = None
    
    def on_created(self, event: FileSystemEvent) -> None:
        """
        Handle file creation event
        
        Args:
            event: File system event
        """
        if event.is_directory:
            return
            
        if not self._should_process_file(event.src_path):
            return
            
        logger.debug(f"File created: {event.src_path}")
        with self.lock:
            self.pending_updates[event.src_path] = "created"
            self._schedule_processing()
    
    def on_modified(self, event: FileSystemEvent) -> None:
        """
        Handle file modification event
        
        Args:
            event: File system event
        """
        if event.is_directory:
            return
            
        if not self._should_process_file(event.src_path):
            return
            
        logger.debug(f"File modified: {event.src_path}")
        with self.lock:
            self.pending_updates[event.src_path] = "modified"
            self._schedule_processing()
    
    def on_deleted(self, event: FileSystemEvent) -> None:
        """
        Handle file deletion event
        
        Args:
            event: File system event
        """
        if event.is_directory:
            return
            
        if not self._should_process_file(event.src_path):
            return
            
        logger.debug(f"File deleted: {event.src_path}")
        with self.lock:
            # If the file was pending an update, remove it
            if event.src_path in self.pending_updates:
                del self.pending_updates[event.src_path]
                
            # Add to deletions
            self.pending_deletions.add(event.src_path)
            self._schedule_processing()
    
    def on_moved(self, event: FileMovedEvent) -> None:
        """
        Handle file move event
        
        Args:
            event: File moved event
        """
        # Handle source file removal
        if not event.is_directory and self._should_process_file(event.src_path):
            logger.debug(f"File moved from: {event.src_path}")
            with self.lock:
                # If the file was pending an update, remove it
                if event.src_path in self.pending_updates:
                    del self.pending_updates[event.src_path]
                    
                # Add to deletions
                self.pending_deletions.add(event.src_path)
        
        # Handle destination file addition
        if not event.is_directory and self._should_process_file(event.dest_path):
            logger.debug(f"File moved to: {event.dest_path}")
            with self.lock:
                self.pending_updates[event.dest_path] = "created"
                
        self._schedule_processing()
    
    def _should_process_file(self, file_path: str) -> bool:
        """
        Determine if a file should be processed based on its extension
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file should be processed
        """
        if not self.file_extensions:
            return True
            
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.file_extensions
    
    def _schedule_processing(self) -> None:
        """
        Schedule delayed processing of file changes
        """
        # Cancel existing timer if any
        if self.timer is not None:
            self.timer.cancel()
            
        # Create a new timer
        self.timer = threading.Timer(self.process_delay, self._process_changes)
        self.timer.daemon = True
        self.timer.start()
    
    def _process_changes(self) -> None:
        """
        Process all pending file changes
        """
        updates = {}
        deletions = set()
        
        # Get a snapshot of pending changes
        with self.lock:
            updates = self.pending_updates.copy()
            deletions = self.pending_deletions.copy()
            self.pending_updates.clear()
            self.pending_deletions.clear()
        
        if not updates and not deletions:
            return
            
        logger.info(f"Processing {len(updates)} updates and {len(deletions)} deletions")
        
        # Process file updates
        for file_path, change_type in updates.items():
            try:
                # Use process_file method if it exists
                if hasattr(self.flow_controller, 'process_file'):
                    self.flow_controller.process_file(file_path, force_reprocess=True)  # type: ignore
                    logger.info(f"Processed {change_type} file: {file_path}")
                else:
                    logger.warning("FlowController does not have process_file method, skipping update")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        # Process file deletions (if index supports removal)
        if hasattr(self.flow_controller, 'remove_from_index') and deletions:
            try:
                for file_path in deletions:
                    getattr(self.flow_controller, 'remove_from_index')(file_path)
                    logger.info(f"Removed from index: {file_path}")
            except Exception as e:
                logger.error(f"Error removing files from index: {e}")
        
        # Save the updated index
        try:
            self.flow_controller.save_index()
            logger.info("Index saved after hot update")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
        
        # Call the update callback if provided
        if self.on_update_callback:
            try:
                self.on_update_callback(updates, deletions)
            except Exception as e:
                logger.error(f"Error in update callback: {e}")


class HotUpdateManager:
    """
    Manager for hot-updating the index when files change
    """
    
    def __init__(
        self,
        flow_controller,
        watch_paths: List[str],
        file_extensions: Optional[List[str]] = None,
        process_delay: float = 1.0,
        recursive: bool = True,
        on_update_callback: Optional[Callable] = None
    ):
        """
        Initialize the hot update manager
        
        Args:
            flow_controller: FlowController instance
            watch_paths: List of paths to watch for changes
            file_extensions: List of file extensions to monitor (e.g., ['.txt', '.md'])
            process_delay: Delay in seconds before processing changes
            recursive: Whether to watch subdirectories recursively
            on_update_callback: Optional callback function to call after processing
        """
        self.flow_controller = flow_controller
        self.watch_paths = [Path(p) for p in watch_paths]
        self.file_extensions = file_extensions or []
        self.process_delay = process_delay
        self.recursive = recursive
        self.on_update_callback = on_update_callback
        
        # Validate watch paths
        for path in self.watch_paths:
            if not path.exists():
                raise ValidationError(f"Watch path does not exist: {path}")
        
        # Create event handler and observer
        self.event_handler = FileChangeHandler(
            flow_controller=flow_controller,
            file_extensions=self.file_extensions,
            process_delay=process_delay,
            on_update_callback=on_update_callback
        )
        
        self.observer = Observer()
        for path in self.watch_paths:
            self.observer.schedule(self.event_handler, str(path), recursive=recursive)
    
    def start(self) -> None:
        """
        Start watching for file changes
        """
        if self.observer.is_alive():
            logger.warning("Hot update observer is already running")
            return
            
        logger.info(f"Starting hot update watcher for paths: {', '.join(str(p) for p in self.watch_paths)}")
        self.observer.start()
    
    def stop(self) -> None:
        """
        Stop watching for file changes
        """
        if not self.observer.is_alive():
            logger.warning("Hot update observer is not running")
            return
            
        logger.info("Stopping hot update watcher")
        self.observer.stop()
        self.observer.join()
    
    def is_watching(self) -> bool:
        """
        Check if the watcher is active
        
        Returns:
            True if the observer is running
        """
        return self.observer.is_alive()
    
    def add_watch_path(self, path: Union[str, Path], recursive: Optional[bool] = None) -> None:
        """
        Add a new path to watch
        
        Args:
            path: Path to watch
            recursive: Whether to watch subdirectories recursively
        """
        path = Path(path)
        if not path.exists():
            raise ValidationError(f"Watch path does not exist: {path}")
            
        if path in self.watch_paths:
            logger.warning(f"Path is already being watched: {path}")
            return
            
        recursive = self.recursive if recursive is None else recursive
        self.observer.schedule(self.event_handler, str(path), recursive=recursive)
        self.watch_paths.append(path)
        logger.info(f"Added watch path: {path}")
    
    def remove_watch_path(self, path: Union[str, Path]) -> None:
        """
        Remove a path from watching
        
        Args:
            path: Path to stop watching
        """
        path = Path(path)
        if path not in self.watch_paths:
            logger.warning(f"Path is not being watched: {path}")
            return
        
        # This is tricky since Observer doesn't provide a direct way to unschedule by path
        # We need to find all watch instances for this path and unschedule them
        watches_to_remove = []
        for watch_key, watch in self.observer._watches.items():  # type: ignore
            if Path(watch.path) == path:
                watches_to_remove.append(watch)
                
        for watch in watches_to_remove:
            self.observer.unschedule(watch)
            
        self.watch_paths.remove(path)
        logger.info(f"Removed watch path: {path}") 