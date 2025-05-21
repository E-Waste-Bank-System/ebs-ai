import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    logger.warning("Google Cloud Storage package not available. Cloud storage features will be disabled.")

class CloudStorage:
    def __init__(self):
        self.enabled = GCS_AVAILABLE and os.environ.get('ENABLE_GCS', 'false').lower() == 'true'
        self.bucket_name = os.environ.get('GCS_BUCKET', '')
        
        if self.enabled:
            try:
                self.client = storage.Client()
                self.bucket = self.client.bucket(self.bucket_name)
                logger.info(f"GCS integration enabled with bucket: {self.bucket_name}")
            except Exception as e:
                logger.error(f"Failed to initialize GCS: {e}")
                self.enabled = False
        else:
            logger.info("GCS integration disabled")
    
    def upload_detection_result(self, local_file_path, run_id):
        if not self.enabled or not os.path.exists(local_file_path):
            return None
        
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            destination_blob_name = f"detections/{today}/{run_id}/{os.path.basename(local_file_path)}"
            
            blob = self.bucket.blob(destination_blob_name)
            blob.upload_from_filename(local_file_path)
            
            blob.make_public()
            
            public_url = blob.public_url
            logger.info(f"Uploaded detection result to GCS: {destination_blob_name}")
            return public_url
        
        except Exception as e:
            logger.error(f"Failed to upload to GCS: {e}")
            return None