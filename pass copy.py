#!/usr/bin/env python3
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os

# Configuration
# General paths to monitor
GENERAL_PATH_1 = '/path/to/scripts/'
GENERAL_PATH_2 = '/another/path/to/scripts/'

EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS', 'EMAIL_PLACEHOLDER')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', 'PASSWORD_PLACEHOLDER')
TO_EMAIL_ADDRESS = os.getenv('TO_EMAIL_ADDRESS', 'EMAIL_PLACEHOLDER')

class LogHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.startswith(GENERAL_PATH_1) or event.src_path.startswith(GENERAL_PATH_2):
            self.send_alert_email(event.src_path)

    def send_alert_email(self, modified_file):
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = TO_EMAIL_ADDRESS
        msg['Subject'] = 'Alert: File Modification Detected'

        body = f"The following file has been modified:\n\n{modified_file}"
        msg.attach(MIMEText(body, 'plain'))

        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            text = msg.as_string()
            server.sendmail(EMAIL_ADDRESS, TO_EMAIL_ADDRESS, text)
            server.quit()
            print(f"Alert email sent successfully for {modified_file}")
        except Exception as e:
            print(f"Failed to send alert email: {e}")

if __name__ == "__main__":
    event_handler = LogHandler()
    observer = Observer()
    observer.schedule(event_handler, path=GENERAL_PATH_1, recursive=True)
    observer.schedule(event_handler, path=GENERAL_PATH_2, recursive=True)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
