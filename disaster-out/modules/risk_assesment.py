import numpy as np

class RiskAssessmentEngine:
    """Aggregates susceptibility & rainfall metrics and evaluates risk for region grids and Lat/Long coordinates."""

    def __init__(self):
        self.regions = [
            {"id": "VIL-001", "name": "Aluva East", "district": "Ernakulam", "bounds": (0, 20, 0, 20)},
            {"id": "VIL-002", "name": "Kuttanad North", "district": "Alappuzha", "bounds": (20, 45, 0, 30)},
            {"id": "VIL-003", "name": "Wayanad Valley", "district": "Wayanad", "bounds": (10, 35, 30, 60)},
            {"id": "VIL-004", "name": "Silchar Central", "district": "Cachar", "bounds": (40, 64, 20, 50)},
            {"id": "VIL-005", "name": "Tezpur Bank", "district": "Sonitpur", "bounds": (30, 60, 40, 64)},
        ]

    @staticmethod
    def detect_language_from_coords(lat: float, lon: float) -> str:
        """Maps Latitude and Longitude to Indian regional languages based on geographic boundaries."""
        if 8.0 <= lat <= 12.8 and 74.5 <= lon <= 77.5:
            return "Malayalam"
        elif 8.0 <= lat <= 13.8 and 76.5 <= lon <= 80.5:
            return "Tamil"
        elif 12.5 <= lat <= 19.8 and 76.8 <= lon <= 84.8:
            return "Telugu"
        elif 21.0 <= lat <= 27.3 and 85.5 <= lon <= 89.8:
            return "Bengali"
        elif 15.6 <= lat <= 22.0 and 72.6 <= lon <= 80.9:
            return "Marathi"
        elif 20.1 <= lat <= 24.7 and 68.1 <= lon <= 74.5:
            return "Gujarati"
        elif 11.5 <= lat <= 18.5 and 74.0 <= lon <= 78.5:
            return "Kannada"
        else:
            return "Hindi"

    def Evaluate_risk(self, susceptibility_map: np.ndarray, rainfall_forecast: np.ndarray) -> list:
        """Evaluates flood risk across administrative region grid boundaries."""
        results = []
        for reg in self.regions:
            r_min, r_max, c_min, c_max = reg["bounds"]
            s_crop = susceptibility_map[r_min:r_max, c_min:c_max]
            rf_crop = rainfall_forecast[r_min:r_max, c_min:c_max]
            
            avg_susceptibility = float(np.mean(s_crop))
            max_rainfall = float(np.max(rf_crop))
            
            risk_score = (avg_susceptibility * 0.6) + (min(max_rainfall, 100.0) / 100.0 * 0.4)
            
            if risk_score > 0.70 or max_rainfall > 70:
                level = "CRITICAL"
                time_window = "3 - 6 Hours"
                color = "red"
            elif risk_score > 0.50 or max_rainfall > 40:
                level = "HIGH"
                time_window = "6 - 12 Hours"
                color = "orange"
            elif risk_score > 0.30:
                level = "MEDIUM"
                time_window = "12 - 24 Hours"
                color = "yellow"
            else:
                level = "LOW"
                time_window = "24+ Hours"
                color = "green"
                
            results.append({
                "region_id": reg["id"],
                "village": reg["name"],
                "district": reg["district"],
                "risk_score": round(risk_score, 2),
                "risk_level": level,
                "max_rainfall_mm": round(max_rainfall, 1),
                "susceptibility_idx": round(avg_susceptibility, 2),
                "lead_time": time_window,
                "color": color
            })
            
        return results

    def evaluate_risk(self, susceptibility_map: np.ndarray, rainfall_forecast: np.ndarray) -> list:
        """Lowercase alias for Evaluate_risk."""
        return self.Evaluate_risk(susceptibility_map, rainfall_forecast)

    def evaluate_custom_location(
        self, 
        lat: float, 
        lon: float, 
        location_name: str, 
        susceptibility_map: np.ndarray, 
        rainfall_forecast: np.ndarray
    ) -> dict:
        """Evaluates flood risk for custom Lat/Long & uploaded image input."""
        avg_susceptibility = float(np.mean(susceptibility_map))
        max_rainfall = float(np.max(rainfall_forecast))
        
        risk_score = (avg_susceptibility * 0.6) + (min(max_rainfall, 100.0) / 100.0 * 0.4)
        
        if risk_score > 0.65 or max_rainfall > 65:
            level = "CRITICAL"
            time_window = "3 - 6 Hours"
        elif risk_score > 0.45 or max_rainfall > 35:
            level = "HIGH"
            time_window = "6 - 12 Hours"
        elif risk_score > 0.25:
            level = "MEDIUM"
            time_window = "12 - 24 Hours"
        else:
            level = "LOW"
            time_window = "24+ Hours"

        auto_lang = self.detect_language_from_coords(lat, lon)

        return {
            "region_id": "CUSTOM-GPS",
            "village": location_name,
            "district": f"Coordinates ({lat:.4f}, {lon:.4f})",
            "lat": lat,
            "lon": lon,
            "risk_score": round(risk_score, 2),
            "risk_level": level,
            "max_rainfall_mm": round(max_rainfall, 1),
            "susceptibility_idx": round(avg_susceptibility, 2),
            "lead_time": time_window,
            "detected_language": auto_lang
        }