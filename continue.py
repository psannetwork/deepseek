#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Continue.py - Orchestrator script to start DeepSeek API Server and LiteLLM proxy for Continue/Cline.
'''

import subprocess
import sys
import os
import time

def main():
    host = "0.0.0.0"
    port = 1234

    print(f"Starting DeepSeek API Server on {host}:{port}...")
    server_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py")
    server_cmd = [sys.executable, server_script, "--host", host, "--port", str(port)]
    
    server_process = subprocess.Popen(server_cmd)
    print(f"Server started with PID: {server_process.pid}")

    # Wait a moment for server to initialize
    time.sleep(2)

    print("Starting LiteLLM proxy with litellm.yaml...")
    litellm_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "litellm.yaml")
    litellm_cmd = ["litellm", "--config", litellm_config]
    
    litellm_process = None
    try:
        litellm_process = subprocess.Popen(litellm_cmd)
        print(f"LiteLLM started with PID: {litellm_process.pid}")
    except FileNotFoundError:
        print("Warning: 'litellm' command not found. Trying python -m litellm...")
        try:
            litellm_cmd = [sys.executable, "-m", "litellm", "--config", litellm_config]
            litellm_process = subprocess.Popen(litellm_cmd)
            print(f"LiteLLM started with PID: {litellm_process.pid}")
        except Exception as e:
            print(f"Error starting LiteLLM: {e}")

    print("Press Ctrl+C to stop all services.")
    
    try:
        while True:
            time.sleep(1)
            if server_process.poll() is not None:
                print("Server process stopped unexpectedly.")
                break
            if litellm_process and litellm_process.poll() is not None:
                print("LiteLLM process stopped unexpectedly.")
                break
    except KeyboardInterrupt:
        print("\nStopping services...")
        if server_process and server_process.poll() is None:
            server_process.terminate()
            server_process.wait()
        if litellm_process and litellm_process.poll() is None:
            litellm_process.terminate()
            litellm_process.wait()
        print("All services stopped.")

if __name__ == "__main__":
    main()
