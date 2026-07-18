DEFECT_KNOWLEDGE = {
    "Cracking": {
        "description": "Visible cracks or split regions in the printed part.",
        "possible_causes": [
            "Low nozzle temperature or weak inter-layer bonding",
            "Excessive cooling or drafts around the printer",
            "Material shrinkage during cooling",
        ],
        "corrective_actions": [
            "Increase nozzle temperature slightly within filament limits",
            "Reduce part cooling fan speed for crack-prone materials",
            "Use an enclosure or reduce drafts near the printer",
        ],
    },
    "Layer_shifting": {
        "description": "Printed layers are offset from the intended position.",
        "possible_causes": [
            "Loose belts or pulley slip",
            "Print speed or acceleration too high",
            "Stepper motor skipped steps due to mechanical resistance",
        ],
        "corrective_actions": [
            "Tighten belts and check pulley set screws",
            "Reduce print speed, acceleration, and jerk settings",
            "Inspect axes for friction or cable obstruction",
        ],
    },
    "Off_platform": {
        "description": "The print detaches from the build platform or starts outside the expected area.",
        "possible_causes": [
            "Poor bed adhesion",
            "Incorrect bed leveling or Z-offset",
            "Bed temperature too low for the filament",
        ],
        "corrective_actions": [
            "Re-level the bed and recalibrate Z-offset",
            "Clean the build surface and apply suitable adhesion aid",
            "Increase bed temperature within filament recommendations",
        ],
    },
    "Stringing": {
        "description": "Thin unwanted strands of filament appear between separate printed regions.",
        "possible_causes": [
            "Nozzle temperature too high",
            "Retraction distance or speed too low",
            "Travel speed too low or wet filament",
        ],
        "corrective_actions": [
            "Lower nozzle temperature in small steps",
            "Tune retraction distance and retraction speed",
            "Dry filament and increase travel speed where appropriate",
        ],
    },
    "Warping": {
        "description": "Edges or corners curl upward due to shrinkage and poor adhesion.",
        "possible_causes": [
            "Bed temperature too low",
            "Cooling too strong during early layers",
            "Insufficient first-layer adhesion",
        ],
        "corrective_actions": [
            "Increase bed temperature within filament limits",
            "Use brim/raft or adhesive for difficult prints",
            "Reduce cooling fan during the first layers",
        ],
    },
    "Under_extrusion": {
        "description": "Missing material, gaps, or weak sparse lines caused by insufficient filament flow.",
        "possible_causes": [
            "Nozzle clog or partial blockage",
            "Filament slipping in extruder gear",
            "Printing temperature too low or flow rate too low",
        ],
        "corrective_actions": [
            "Clean nozzle or perform cold pull",
            "Check extruder tension and filament path",
            "Increase nozzle temperature or calibrate extrusion multiplier",
        ],
    },
    "Over_extrusion": {
        "description": "Excess material, thick ridges, blobs, or rough overfilled surfaces.",
        "possible_causes": [
            "Flow rate or extrusion multiplier too high",
            "Filament diameter configured incorrectly",
            "Nozzle temperature too high",
        ],
        "corrective_actions": [
            "Calibrate E-steps and extrusion multiplier",
            "Measure filament diameter and update slicer setting",
            "Lower nozzle temperature slightly",
        ],
    },
    "Nozzle_clog": {
        "description": "Blocked nozzle causing weak extrusion, missing lines, or failed filament flow.",
        "possible_causes": [
            "Debris or burnt filament inside the nozzle",
            "Low-quality or contaminated filament",
            "Heat creep in hotend",
        ],
        "corrective_actions": [
            "Perform cold pull or replace nozzle",
            "Use clean, dry filament",
            "Check hotend cooling fan and heat break",
        ],
    },
}


def recommendation_for(class_name: str) -> dict:
    return DEFECT_KNOWLEDGE.get(
        class_name,
        {
            "description": "No class-specific explanation is available.",
            "possible_causes": ["The defect class is not in the current recommendation table."],
            "corrective_actions": ["Collect labeled examples and add this class before deployment."],
        },
    )
