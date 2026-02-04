AI_ROUTING_MAP = {
    "CT": ["lung_nodule", "brain_hemorrhage"],
    "MR": ["brain_tumor"],
    "XRAY": ["chest_abnormality"],
    "US": ["abdomen_findings"]
}

def route_to_ai_models(modality):
    return AI_ROUTING_MAP.get(modality, [])
