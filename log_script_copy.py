#!/usr/bin/env python3
import os
import subprocess
import datetime
import psutil
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Hardcoded app-specific password
APP_SPECIFIC_PASSWORD = 'PASSWORD HERE'  # Replace with your actual app-specific password

def get_battery_percentage():
    try:
        process = subprocess.Popen(["pmset", "-g", "batt"], stdout=subprocess.PIPE)
        output, _ = process.communicate()
        output = output.decode("utf-8")
        start = output.find('InternalBattery-0') + 17
        end = output.find('%', start)
        return output[start:end].strip()
    except Exception as e:
        return f"Error: {e}"

def get_network_connections():
    try:
        process = subprocess.Popen(["networksetup", "-listallhardwareports"], stdout=subprocess.PIPE)
        output, _ = process.communicate()
        return output.decode("utf-8")
    except Exception as e:
        return f"Error: {e}"

def get_external_devices():
    try:
        process = subprocess.Popen(["system_profiler", "SPUSBDataType"], stdout=subprocess.PIPE)
        output, _ = process.communicate()
        return output.decode("utf-8")
    except Exception as e:
        return f"Error: {e}"

def get_system_performance():
    try:
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        return cpu_usage, memory_usage
    except Exception as e:
        return f"Error: {e}"

def get_disk_usage():
    try:
        disk_usage = psutil.disk_usage('/').percent
        return f"Disk Usage: {disk_usage}%"
    except Exception as e:
        return f"Error: {e}"

def get_running_apps():
    try:
        process = subprocess.Popen(["osascript", "-e", 'tell application "System Events" to get the name of every process whose background only is false'], stdout=subprocess.PIPE)
        output, _ = process.communicate()
        return output.decode("utf-8")
    except Exception as e:
        return f"Error: {e}"

def perform_security_check():
    try:
        process = subprocess.Popen(["/opt/homebrew/bin/nmap", "your own here"], stdout=subprocess.PIPE)
        output, _ = process.communicate()
        return output.decode("utf-8")
    except Exception as e:
        return f"Error: {e}"

def send_email(subject, body, to_email, from_email, app_specific_password):
    try:
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'html'))  
        server = smtplib.SMTP('smtp.mail.me.com', 587)
        server.starttls()
        server.login(from_email, app_specific_password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
    except Exception as e:
        print(f"Failed to send email: {e}")

def get_system_temperature():
    try:
        process = subprocess.Popen(["osx-cpu-temp"], stdout=subprocess.PIPE)
        output, _ = process.communicate()
        return output.decode('utf-8').strip()
    except Exception as e:
        return f"Error: {e}"

def check_software_updates():
    try:
        process = subprocess.Popen(["softwareupdate", "-l"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, _ = process.communicate()
        return output.decode("utf-8").strip()
    except Exception as e:
        return f"Error: {e}"

def get_wifi_details():
    try:
        process = subprocess.Popen(["/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport", "-I"], stdout=subprocess.PIPE)
        output, _ = process.communicate()
        return output.decode("utf-8").strip()
    except Exception as e:
        return f"Error: {e}"

def get_firewall_status():
    try:
        process = subprocess.Popen(["/usr/libexec/ApplicationFirewall/socketfilterfw", "--getglobalstate"], stdout=subprocess.PIPE)
        output, _ = process.communicate()
        return output.decode("utf-8").strip()
    except Exception as e:
        return f"Error: {e}"

def get_system_uptime():
    try:
        process = subprocess.Popen(['uptime'], stdout=subprocess.PIPE)
        output, _ = process.communicate()
        return output.decode('utf-8').strip()
    except Exception as e:
        return f"Error: {e}"

def track_cpu_gpu_load():
    try:
        cpu_load = psutil.cpu_percent(interval=1)
        gpu_load = "GPU load tracking feature not implemented."
        return f"CPU Load: {cpu_load}%, {gpu_load}"
    except Exception as e:
        return f"Error: {e}"

def get_cpu_temperature():
    try:
        temps = psutil.sensors_temperatures()
        if "coretemp" in temps:
            cpu_temp = temps["coretemp"][0].current
            return f"CPU Temperature: {cpu_temp}°C"
        else:
            return "CPU temperature sensors not available."
    except Exception as e:
        return f"Error retrieving CPU temperature: {e}"

def get_running_services():
    try:
        process = subprocess.Popen(["service", "--status-all"], stdout=subprocess.PIPE)
        output, _ = process.communicate()
        return output.decode("utf-8")
    except Exception as e:
        return f"Error retrieving running services: {e}"

def get_detailed_network_info():
    try:
        process = subprocess.Popen(["ifconfig"], stdout=subprocess.PIPE)
        output, _ = process.communicate()
        return output.decode("utf-8")
    except Exception as e:
        return f"Error retrieving network information: {e}"

def create_log_file():
    battery_percentage = get_battery_percentage()
    network_connections = get_network_connections()
    external_devices = get_external_devices()
    cpu_usage, memory_usage = get_system_performance()
    disk_usage = get_disk_usage()
    running_apps = get_running_apps()
    security_check = perform_security_check()
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    system_temperature = get_system_temperature()
    software_updates = check_software_updates()
    wifi_details = get_wifi_details()
    firewall_status = get_firewall_status()
    system_uptime = get_system_uptime()
    cpu_gpu_load = track_cpu_gpu_load()
    cpu_temperature = get_cpu_temperature()
    running_services = get_running_services()
    detailed_network_info = get_detailed_network_info()

    log_data = """
    <html>
    <body>
    <h1>System Log Report</h1>
    <p><strong>Date and Time:</strong> {current_time}</p>
    <p><strong>Battery Information:</strong><br>{battery_percentage}%</p>
    <p><strong>Network Connections:</strong><br>{network_connections}</p>
    <p><strong><b>External Devices Connected:</b></strong><br>{external_devices}</p>
    <p><strong>System Performance Metrics:</strong><br>CPU Usage: {cpu_usage}%<br>Memory Usage: {memory_usage}%<br>Disk Usage: {disk_usage}%</p>
    <p><strong>Currently Running Applications:</strong><br>{running_apps}</p>
    <p><strong>Basic Security Check (Open Ports):</strong><br>{security_check}</p>
    <p><strong><b>System Temperature:</b></strong><br>{system_temperature}</p>
    <p><strong><b>Pending Software Updates:</b></strong><br>{software_updates}</p>
    <p><strong>Wi-Fi Details:</strong><br>{wifi_details}</p>
    <p><strong><b>Firewall Status:</b></strong><br>{firewall_status}</p>
    <p><strong>System Uptime:</strong><br>{system_uptime}</p>
    <p><strong>CPU and GPU Load:</strong><br>{cpu_gpu_load}</p>
    <p><strong>CPU Temperature:</strong><br>{cpu_temperature}</p>
    <p><strong>Running Services:</strong><br>{running_services}</p>
    <p><strong>Detailed Network Info:</strong><br>{detailed_network_info}</p>
    </body>
    </html>
    """.format(
        current_time=current_time,
        battery_percentage=battery_percentage,
        network_connections=network_connections,
        external_devices=external_devices,
        cpu_usage=cpu_usage,
        memory_usage=memory_usage,
        disk_usage=disk_usage,
        running_apps=running_apps,
        security_check=security_check,
        system_temperature=system_temperature,
        software_updates=software_updates,
        wifi_details=wifi_details,
        firewall_status=firewall_status,
        system_uptime=system_uptime,
        cpu_gpu_load=cpu_gpu_load,
        cpu_temperature=cpu_temperature,
        running_services=running_services,
        detailed_network_info=detailed_network_info
    )

    subject = "System Log Report"
    to_email = "EMAIL RECIPIENT HERE"
    from_email = "EMAIL SENDER HERE"

    send_email(subject, log_data, to_email, from_email, APP_SPECIFIC_PASSWORD)

create_log_file()
