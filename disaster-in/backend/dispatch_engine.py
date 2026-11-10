"""
Emergency Dispatch Engine:
Maps Emergency Type + Severity Level + Extracted Details -> Optimal Emergency Units to Deploy.
Calculates:
- Primary & Secondary Agencies (NDRF, SDRF, Fire Brigade, 108 EMS Ambulance, Police QRT, USAR)
- Specialized Units, Quantities, and Call Signs
- Total Responders & Tactical Depot
- Estimated Time of Arrival (ETA)
- Radio Tactical Comms Frequency & Emergency Priority Code
- Critical Equipment Checklist & SOP Operational Directives
"""

import uuid
from models import EmergencyTypeEnum, SeverityLevelEnum, LLMExtractionResult, DispatchPlan, DispatchUnit


class EmergencyDispatchEngine:

    @classmethod
    def generate_dispatch_plan(cls, extraction: LLMExtractionResult) -> DispatchPlan:
        plan_id = f"DISP-{uuid.uuid4().hex[:6].upper()}"
        etype = extraction.emergency_type
        sev = extraction.severity
        trapped = extraction.people_trapped
        injured = extraction.people_injured

        # Determine Priority Level & ETA
        if sev == SeverityLevelEnum.CRITICAL or extraction.severity_score >= 5:
            priority = "PRIORITY-1 (RED ALERT / CODE TACTICAL OMEGA)"
            base_eta = 4
        elif sev == SeverityLevelEnum.HIGH or extraction.severity_score >= 4:
            priority = "PRIORITY-2 (HIGH URGENCY / CODE AMBER)"
            base_eta = 7
        elif sev == SeverityLevelEnum.MEDIUM:
            priority = "PRIORITY-3 (MODERATE RESPONSE / CODE YELLOW)"
            base_eta = 12
        else:
            priority = "PRIORITY-4 (STANDARD DISPATCH / CODE GREEN)"
            base_eta = 18

        # Dispatch Matrix mapping
        units = []
        primary_agency = ""
        backup_agency = ""
        depot = ""
        radio_channel = "VHF 156.800 MHz (TAC-1 EMERGENCY)"
        equipment = []
        protocols = []

        if etype == EmergencyTypeEnum.FLOOD:
            primary_agency = "National Disaster Response Force (NDRF) - 9th Battalion"
            backup_agency = "State Disaster Response Force (SDRF) & Marine Police"
            depot = f"Disaster Response Hub ({extraction.city or 'Regional Center'})"
            radio_channel = "NDRF AQUA-NET 148.550 MHz"

            boat_count = max(2, (trapped // 3) + 1)
            units.append(DispatchUnit(
                unit_type="Inflatable Motorized Rescue Boat (Gemini/Zodiac)",
                quantity=boat_count,
                call_sign="BOAT-RESCUE-ALPHA",
                station=depot,
                equipment=["40HP OBM Engines", "Life Jackets (Type III)", "Throw Lines", "Thermal Searchlights"]
            ))
            units.append(DispatchUnit(
                unit_type="Aquatic Rapid Extraction Team",
                quantity=8 + (boat_count * 2),
                call_sign="DIVER-TASKFORCE-01",
                station=depot,
                equipment=["Wet suits", "Underwater Comms", "Floatable Basket Stretchers"]
            ))
            units.append(DispatchUnit(
                unit_type="High-Capacity Submersible Dewatering Pump Truck",
                quantity=1,
                call_sign="PUMP-UNIT-04",
                station=f"{extraction.city} Municipal Engineering Depot",
                equipment=["5000 LPM High-Pressure Dewatering Hose", "Diesel Generators"]
            ))
            units.append(DispatchUnit(
                unit_type="108 Advanced Life Support (ALS) Ambulance (Flood Config)",
                quantity=max(1, (injured // 2) + 1),
                call_sign="AMBULANCE-ALS-12",
                station=f"{extraction.city} Civil Hospital Triage Depot",
                equipment=["Hypothermia Survival Blankets", "Portable Oxygen", "IV Saline Kits"]
            ))

            equipment = [
                "Motorized Inflatable Boats (Zodiac ZM-400)",
                "High-Buoyancy PFD Life Jackets (SOLAS Certified)",
                "Night-Vision Thermal Imaging Scanners",
                "High-Intensity Battery LED Floodlight Banks",
                "Satellite Phone Comms & Megaphones",
                "Emergency Rations & Potable Drinking Water Packs"
            ]
            protocols = [
                "1. Establish Safe Water Entry Staging Point upstream of submerged debris.",
                "2. Cut off regional electrical grid transformer feeds immediately to prevent electrocution.",
                "3. Prioritize evacuation of pediatric, geriatric, and medically compromised individuals first.",
                "4. Maintain direct VHF relay with Incident Commander at State EOC."
            ]

        elif etype == EmergencyTypeEnum.FIRE:
            primary_agency = "State Fire & Emergency Services Directorate"
            backup_agency = "108 Emergency Medical Services & Police Traffic Command"
            depot = f"{extraction.landmark or extraction.city} Central Fire Station"
            radio_channel = "FIRE-TAC CH-3 154.280 MHz"

            tenders = 2 if sev in [SeverityLevelEnum.CRITICAL, SeverityLevelEnum.HIGH] else 1
            units.append(DispatchUnit(
                unit_type="Heavy Water Tender (10,000 Liters + High-Pressure Monitor)",
                quantity=tenders,
                call_sign="WATER-TENDER-T-01",
                station=depot,
                equipment=["Water Cannon Monitor", "Class A/B Foam Proportioners", "Lay-flat Delivery Hoses"]
            ))
            units.append(DispatchUnit(
                unit_type="Hydraulic Turntable Aerial Ladder Platform (55m Skylift)",
                quantity=1,
                call_sign="AERIAL-LADDER-L-02",
                station=f"{extraction.city} Headquarters Fire Depot",
                equipment=["Elevated Rescue Cage", "Master Stream Nozzle", "Rappelling Harnesses"]
            ))
            units.append(DispatchUnit(
                unit_type="Breathing Apparatus Van & Smoke Extractor Unit",
                quantity=1,
                call_sign="SMOKE-EJECTOR-V-05",
                station=depot,
                equipment=["SCBA 300-Bar Cylinders", "Positive Pressure Ventilation Fans", "Thermal Cameras"]
            ))
            units.append(DispatchUnit(
                unit_type="108 Burn Trauma Care ALS Ambulance",
                quantity=max(2, (injured // 2) + 1),
                call_sign="MED-TRAUMA-BURNS-09",
                station=f"{extraction.city} District Hospital",
                equipment=["Hydrogel Burn Dressings", "Mechanical CPR", "Endotracheal Intubation Kit"]
            ))

            equipment = [
                "Self-Contained Breathing Apparatus (SCBA Draeger 300 bar)",
                "FLIR K55 Thermal Imaging Cameras for structural hot-spot detection",
                "Hydraulic Forcible Entry 'Hooligan' Spreaders",
                "Aqueous Film-Forming Foam (AFFF) 6% Fire Suppressant",
                "High-Volume Positive Pressure Ventilation (PPV) Blowers"
            ]
            protocols = [
                "1. Operator confirmed multi-person entrapment: Initiate 360-degree size-up and aggressive primary search.",
                "2. Ventilate vertical stairwells and secure secondary escape egress.",
                "3. Establish triage perimeter 100m upwind from toxic smoke plume.",
                "4. Hospital notifications initiated for immediate burn ward activation."
            ]

        elif etype == EmergencyTypeEnum.BUILDING_COLLAPSE:
            primary_agency = "Urban Search & Rescue (USAR) Specialist Strike Team"
            backup_agency = "NDRF Heavy Disaster Management Battalion & Civil Defense"
            depot = f"{extraction.city} Disaster Operations Facility"
            radio_channel = "USAR-RESCUE-NET 151.625 MHz"

            units.append(DispatchUnit(
                unit_type="Heavy USAR Technical Rescue Vehicle",
                quantity=1,
                call_sign="USAR-HEAVY-STRIKE-01",
                station=depot,
                equipment=["Hydraulic Concrete Crushers", "Diamond Blade Cutters", "Air Lifting Bags (70 Ton)"]
            ))
            units.append(DispatchUnit(
                unit_type="K9 Canine Search & Acoustic Seismic Detection Squad",
                quantity=1,
                call_sign="K9-SEARCH-SQUAD-3",
                station=f"{extraction.city} USAR Canine Depot",
                equipment=["Geophone Acoustic Listening Sensors", "Fiberoptic Borescope Search Cam", "USAR Dogs"]
            ))
            units.append(DispatchUnit(
                unit_type="Hydraulic Telescopic Crane (50-Ton) & Shoring Unit",
                quantity=1,
                call_sign="CRANE-RIGGER-CR-07",
                station="Municipal Heavy Machinery Yard",
                equipment=["Pneumatic Paratech Shoring Struts", "Timber Cribbing Blocks"]
            ))
            units.append(DispatchUnit(
                unit_type="Disaster Medical Assistance Team (DMAT) Mobile ICU",
                quantity=max(2, (trapped // 2) + 1),
                call_sign="DMAT-MOBILE-ICU-03",
                station=f"{extraction.city} Medical College",
                equipment=["Crush Injury Hyperkalemia Kits", "Field Amputation Surgical Pack", "Blood Warmer"]
            ))

            equipment = [
                "Acoustic Seismic Void Sensors (LifeDetector LD3)",
                "High-Pressure Kevlar Air Lifting Bags (up to 74 Tons)",
                "Diamond-Tipped Petrol Rotary Concrete Saw",
                "Paratech Heavy Pneumatic Rescue Struts",
                "Telescopic High-Definition Borescope Cameras with 2-way audio"
            ]
            protocols = [
                "1. Implement 'Call-Out and Listen' silence protocol across collapse perimeter.",
                "2. Rapid structural stabilization and shoring of compromised walls before void entry.",
                "3. Treat all trapped patients proactively for Crush Injury Syndrome before slab release.",
                "4. Continuous monitoring for gas line leaks and electrical conduits."
            ]

        elif etype == EmergencyTypeEnum.ROAD_ACCIDENT:
            primary_agency = "Highway Patrol Quick Response Team & Trauma Rescue Services"
            backup_agency = "108 Highway Ambulance Network & Heavy Crane Recovery"
            depot = f"{extraction.landmark or extraction.city} Highway Outpost"
            radio_channel = "HIGHWAY-PATROL-VHF 162.400 MHz"

            units.append(DispatchUnit(
                unit_type="Extrication Rescue Tender (Jaws of Life)",
                quantity=1,
                call_sign="EXTRICATION-RESCUE-01",
                station=depot,
                equipment=["Hydraulic Spreaders & Cutters", "Spinal Immobilization Boards", "Ram Extenders"]
            ))
            units.append(DispatchUnit(
                unit_type="Highway Trauma ALS Ambulances",
                quantity=max(2, (injured // 3) + 1),
                call_sign="EMS-HIGHWAY-UNIT-4",
                station=f"{extraction.city} Trauma Care Center",
                equipment=["Cervical Collars", "Traction Splints", "Blood Pressure Monitors", "Portable Suction"]
            ))
            units.append(DispatchUnit(
                unit_type="Police Highway Traffic Cordon & Crash Investigation",
                quantity=2,
                call_sign="HIGHWAY-POLICE-PATROL-11",
                station=depot,
                equipment=["Warning Flare Batteries", "Reflective Cones", "Emergency Detour Signage"]
            ))

            equipment = [
                "Holmatro Hydraulic Rescue Cutters (Jaws of Life)",
                "Kendrick Extrication Devices (KED) & Scoop Stretchers",
                "Highway High-Visibility Illuminated Flare Triangles",
                "Spill Containment Absorbent Pads for Fuel / Oil",
                "Vehicle Battery De-energizing Disconnect Tools"
            ]
            protocols = [
                "1. Close upstream lane 500 meters prior to incident to prevent secondary multi-car collisions.",
                "2. Neutralize vehicle ignition battery to eliminate electrical spark ignition of spilled fuel.",
                "3. Full C-spine immobilization before any patient extrication.",
                "4. Level-1 Trauma Center pre-arrival patch broadcast."
            ]

        elif etype == EmergencyTypeEnum.GAS_LEAK:
            primary_agency = "CBRN Hazmat Incident Response Taskforce"
            backup_agency = "Specialized Chemical Neutralization Unit & District Police"
            depot = f"{extraction.city} Industrial Emergency Center"
            radio_channel = "HAZMAT-OPS 155.050 MHz"

            units.append(DispatchUnit(
                unit_type="Hazmat Level-A Chemical Containment Vehicle",
                quantity=1,
                call_sign="HAZMAT-ALPHA-01",
                station=depot,
                equipment=["Encapsulated Level-A Suits", "Multi-Gas Detectors (PID)", "Chlorine/Ammonia Capping Kits"]
            ))
            units.append(DispatchUnit(
                unit_type="Decontamination & Evacuation Bus",
                quantity=1,
                call_sign="DECON-UNIT-02",
                station=depot,
                equipment=["Decon Shower Tents", "Neutralizing Solution", "Disposable Coveralls"]
            ))
            units.append(DispatchUnit(
                unit_type="Specialized Respiratory Care Ambulance",
                quantity=2,
                call_sign="EMS-RESP-CARE-06",
                station=f"{extraction.city} Pulmonary Care Hospital",
                equipment=["High-Flow Bronchodilators", "Nebulizers", "Oxygen Manifolds"]
            ))

            equipment = [
                "RAE Systems MultiRAE 5-Gas Environmental Analyzers",
                "Fully Encapsulated Vapor-Protective Level-A Hazardous Suits",
                "Emergency Pipe Clamps and Magnetic Tank Seal Patches",
                "Portable Hazardous Decontamination Shower Systems",
                "Explosion-Proof Intrinsically Safe Exhaust Blowers"
            ]
            protocols = [
                "1. Approach strictly UPWIND and UPGRADE from leak source.",
                "2. Establish 1000m Hot Zone, Warm Zone (Decon), and Cold Zone.",
                "3. Broadcast automated evacuation sirens and emergency cell alerts for nearby populace.",
                "4. Coordinate with factory engineering leads for main valve isolation."
            ]

        else:
            # Default / Medical / Crime
            primary_agency = "Emergency Medical & Police Combined Dispatch"
            backup_agency = "Civil Protection Taskforce"
            depot = f"{extraction.city} Central Emergency Depot"
            radio_channel = "DISPATCH-COMM-1 150.000 MHz"

            units.append(DispatchUnit(
                unit_type="Advanced Life Support (ALS) Emergency Ambulance",
                quantity=2,
                call_sign="AMBULANCE-ALS-01",
                station=depot,
                equipment=["Cardiac Monitor / Defibrillator", "Oxygen", "Trauma Bag"]
            ))
            units.append(DispatchUnit(
                unit_type="Police Quick Reaction Team (QRT)",
                quantity=1,
                call_sign="POLICE-QRT-05",
                station=f"{extraction.city} Police Station",
                equipment=["Tactical Gear", "Area Cordon Tape", "First Aid"]
            ))

            equipment = [
                "Zoll X Series Defibrillator / Cardiac Monitor",
                "Emergency Oxygen Cylinders & BVM",
                "Tactical First Response First Aid Kit",
                "Automated External Defibrillator (AED)"
            ]
            protocols = [
                "1. Dispatch immediate first responders with lights and sirens.",
                "2. Continuous radio coordination with field triage commander.",
                "3. Alert receiving hospital emergency department."
            ]

        total_responders = sum(u.quantity * (3 if "Ambulance" in u.unit_type else 4) for u in units)

        return DispatchPlan(
            plan_id=plan_id,
            priority_level=priority,
            emergency_type=etype.value,
            primary_agency=primary_agency,
            backup_agency=backup_agency,
            recommended_units=units,
            total_responders=max(8, total_responders),
            estimated_eta_minutes=base_eta,
            radio_channel=radio_channel,
            depot_assigned=depot,
            equipment_checklist=equipment,
            critical_protocols=protocols
        )
