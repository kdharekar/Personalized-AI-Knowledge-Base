import csv
import os
from datetime import datetime

class FeedbackLogger:
    """
    A service to log user feedback for search results into a CSV file.
    """
    def __init__(self, filepath="feedback.csv"):
        self.filepath = filepath
        self.fieldnames = [
            "timestamp", 
            "search_id", 
            "rating", 
            "query", 
            "answer", 
            "comment"
        ]
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Creates the CSV file with a header row if it doesn't exist."""
        if not os.path.exists(self.filepath):
            with open(self.filepath, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
                print(f"Feedback log file created at: {self.filepath}")

    def log(self, feedback_data: dict):
        """
        Appends a new row of feedback to the CSV file.

        Args:
            feedback_data (dict): A dictionary containing feedback details.
        """
        feedback_data['timestamp'] = datetime.now().isoformat()
        
        try:
            with open(self.filepath, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow(feedback_data)
            print(f"Successfully logged feedback for search_id: {feedback_data['search_id']}")
        except Exception as e:
            print(f"Error logging feedback: {e}")