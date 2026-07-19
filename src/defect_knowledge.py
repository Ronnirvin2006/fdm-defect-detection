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
    "Blobs_and_zits": {
        "description": "Small bumps, zits, or excess plastic deposits on the print surface.",
        "possible_causes": [
            "Retraction restart distance too high",
            "Pressure build-up in the nozzle",
            "Very small perimeter moves or unstable extrusion flow",
        ],
        "corrective_actions": [
            "Tune retraction and extra restart distance",
            "Enable coasting or pressure advance if available",
            "Reduce nozzle temperature slightly and calibrate flow",
        ],
    },
    "Bed_adhesion_failure": {
        "description": "The first layer does not stick correctly to the build plate.",
        "possible_causes": [
            "Dirty bed surface",
            "Incorrect Z-offset",
            "Bed temperature or first-layer speed not suitable",
        ],
        "corrective_actions": [
            "Clean the bed and re-level it",
            "Adjust Z-offset for better first-layer squish",
            "Increase first-layer bed temperature or slow first layer",
        ],
    },
    "Blob_of_death": {
        "description": "Large mass of plastic builds up around the nozzle or print.",
        "possible_causes": [
            "Print detached and filament kept extruding",
            "Severe nozzle leak",
            "Unattended extrusion failure",
        ],
        "corrective_actions": [
            "Stop the print immediately",
            "Heat the hotend carefully before removing plastic",
            "Inspect nozzle, heat block, and bed adhesion before restarting",
        ],
    },
    "Layer_separation": {
        "description": "Adjacent layers separate or delaminate from each other.",
        "possible_causes": [
            "Nozzle temperature too low",
            "Cooling too strong",
            "Material shrinkage or poor layer bonding",
        ],
        "corrective_actions": [
            "Increase nozzle temperature",
            "Reduce cooling fan speed",
            "Use an enclosure for shrink-prone materials",
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
    "No_defect": {
        "description": "The image appears to show a normal print without a known defect.",
        "possible_causes": [
            "No defect is detected by the trained classifier.",
        ],
        "corrective_actions": [
            "Continue monitoring the print.",
            "Inspect confidence scores if the prediction is uncertain.",
        ],
    },
    "Spaghetti": {
        "description": "Loose tangled filament appears because the print failed and extrusion continued.",
        "possible_causes": [
            "Print detached from the bed",
            "Severe layer shift or support failure",
            "Nozzle printing in air after part failure",
        ],
        "corrective_actions": [
            "Stop the print and clear loose filament",
            "Improve bed adhesion and support settings",
            "Check slicer preview and reduce risky overhangs",
        ],
    },
    "Z_banding": {
        "description": "Repeating horizontal bands appear along the Z direction.",
        "possible_causes": [
            "Z-axis lead screw wobble or binding",
            "Inconsistent extrusion",
            "Frame vibration or mechanical misalignment",
        ],
        "corrective_actions": [
            "Inspect and clean Z rods or lead screws",
            "Check couplers, wheels, and frame alignment",
            "Calibrate extrusion and reduce vibration",
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
