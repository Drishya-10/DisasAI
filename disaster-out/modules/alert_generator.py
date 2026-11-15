class MultiModeAlertGenerator:
    """Generates structured emergency advisory text."""

    @staticmethod
    def generate_template_alert(risk_data: dict) -> str:
        """Deterministic template fallback."""
        village = risk_data["village"]
        district = risk_data["district"]
        level = risk_data["risk_level"]
        lead_time = risk_data["lead_time"]
        rainfall = risk_data["max_rainfall_mm"]

        if level in ["CRITICAL", "HIGH"]:
            action = "Evacuate low-lying areas immediately to designated relief shelters. Move livestock and valuables to high ground."
        elif level == "MEDIUM":
            action = "Stay on high alert. Avoid crossing rivers, stream banks, and flooded roads. Store drinking water."
        else:
            action = "Monitor local weather reports and stay informed of further updates."

        return (
            f"EMERGENCY FLOOD ALERT [{level}]\n"
            f"Location: {village}, District: {district}\n"
            f"Expected Lead Time: Within {lead_time}\n"
            f"Forecast Rainfall: {rainfall} mm/hr\n"
            f"Advisory: {action}"
        )

    @staticmethod
    def generate_llm_alert(risk_data: dict, api_key: str = None) -> str:
        """LLM generator (simulated or active API wrapper)."""
        if not api_key:
            # Fallback to smart structured generation if API key is not supplied
            return MultiModeAlertGenerator.generate_template_alert(risk_data)
        
        # Example API Prompt pattern (OpenAI / Gemini integration template)
        prompt = f"""
        Act as an official Disaster Management Officer. Write a concise, urgent SMS alert (< 60 words) for citizens.
        Details: Village: {risk_data['village']}, Risk: {risk_data['risk_level']}, Time Window: {risk_data['lead_time']}, Expected Rain: {risk_data['max_rainfall_mm']} mm/hr.
        Provide clear safety actions.
        """
        # API call implementation goes here...
        return MultiModeAlertGenerator.generate_template_alert(risk_data)