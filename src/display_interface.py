import os
import time

class AlertDisplay:
    def __init__(self, mode='console'):
        """
        Manages the visual interface for the driver. 
        In a production environment, 'mode' could be set to 'oled' using libraries like luma.oled.
        """
        self.mode = mode

    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def update_dashboard(self, hazards, fps):
        """Renders the current state of detected hazards to the driver."""
        if self.mode == 'console':
            self.clear_screen()
            print("="*40)
            print("🚗 SMART VEHICLE HAZARD DASHBOARD 🚗")
            print("="*40)
            print(f"System Status: ONLINE | Speed: {fps:.1f} FPS\n")
            
            if not hazards:
                print("✅ Road Clear. No immediate hazards detected.")
            else:
                print("⚠️ HAZARDS DETECTED:")
                for i, h in enumerate(hazards):
                    warning_symbol = "🔴" if h['priority'] == 'CRITICAL' else "🟠"
                    print(f"  {warning_symbol} [{h['priority']}] {h['class'].upper()} (Conf: {h['confidence']})")
            
            print("\n" + "="*40)

if __name__ == "__main__":
    display = AlertDisplay()
    # Mock update
    display.update_dashboard([{'class': 'person', 'priority': 'CRITICAL', 'confidence': 0.88}], 34.5)
