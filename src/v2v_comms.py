import socketio
import paho.mqtt.client as mqtt
import json
import time
import threading

class V2VCommunicationStack:
    def __init__(self, ws_server_ip='192.168.1.100', mqtt_broker_ip='192.168.1.100'):
        # WebSocket Setup (Urgent Alerts)
        self.sio = socketio.Client()
        self.ws_url = f"http://{ws_server_ip}:5000"
        
        # MQTT Setup (Telemetry & Logging)
        self.mqtt_client = mqtt.Client("Vehicle_Node_1")
        self.mqtt_broker = mqtt_broker_ip
        
        self._setup_callbacks()

    def _setup_callbacks(self):
        @self.sio.event
        def connect():
            print("Connected to V2V WebSocket Server")

        @self.sio.event
        def disconnect():
            print("Disconnected from V2V Server")

        @self.sio.on('incoming_alert')
        def on_alert(data):
            print(f"[V2V ALERT RECEIVE] Priority: {data['priority']} | Hazard: {data['class']}")

    def connect_all(self):
        try:
            self.sio.connect(self.ws_url)
        except Exception as e:
            print(f"WebSocket Connection Failed: {e}")
            
        try:
            self.mqtt_client.connect(self.mqtt_broker, 1883, 60)
            self.mqtt_client.loop_start()
        except Exception as e:
            print(f"MQTT Connection Failed: {e}")

    def broadcast_hazard(self, hazard_data):
        """Sends urgent data via WebSocket and logs to MQTT"""
        payload = {
            'timestamp': time.time(),
            'class': hazard_data['class'],
            'priority': hazard_data['priority'],
            'confidence': hazard_data['confidence']
        }
        
        # 1. Low-Latency Transmission
        if self.sio.connected:
            self.sio.emit('hazard_alert', payload)
            
        # 2. Asynchronous Logging
        self.mqtt_client.publish("v2v/telemetry/hazards", json.dumps(payload))
        
    def disconnect(self):
        self.sio.disconnect()
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()

if __name__ == "__main__":
    # Test script for communication stack
    comms = V2VCommunicationStack(ws_server_ip='localhost', mqtt_broker_ip='localhost')
    # comms.connect_all()
    print("Communication module initialized.")
