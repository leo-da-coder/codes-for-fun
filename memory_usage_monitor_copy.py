#!/opt/homebrew/bin/python3
import subprocess
import re

def get_memory_usage():
    vm_stat = subprocess.run(['vm_stat'], capture_output=True, text=True).stdout
    stats = {line.split(':')[0].rstrip('.'): int(re.findall(r'\d+', line.split(':')[1])[0])
             for line in vm_stat.splitlines() if ':' in line}
    
    page_size_output = subprocess.run(['sysctl', 'hw.pagesize'], capture_output=True, text=True).stdout
    page_size = int(re.findall(r'\d+', page_size_output)[0])

    used_memory_pages = stats.get('Pages active', 0) + stats.get('Pages wired down', 0)
    used_memory_bytes = used_memory_pages * page_size
    
    free_memory_pages = stats.get('Pages free', 0) + stats.get('Pages inactive', 0)
    free_memory_bytes = free_memory_pages * page_size
    
    total_memory_bytes = used_memory_bytes + free_memory_bytes
    total_memory_gb = total_memory_bytes / (1024 ** 3)
    used_memory_gb = used_memory_bytes / (1024 ** 3)
    
    memory_usage_percent = (used_memory_gb / total_memory_gb) * 100
    
    return memory_usage_percent

def send_notification(usage_percent):
    title = "Memory Usage Alert"
    text = f"Your system is using {usage_percent:.2f}% of its memory."
    script = f'display notification "{text}" with title "{title}"'
    subprocess.run(['osascript', '-e', script])

if __name__ == "__main__":
    memory_usage_percent = get_memory_usage()
    if memory_usage_percent > 80:  # You can adjust the threshold here
        send_notification(memory_usage_percent)
