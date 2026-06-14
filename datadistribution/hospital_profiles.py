"""
hospital_profiles.py
====================
Defines the 10 named hospital archetypes (NO fixed w_k values here).
Each archetype contains a qualitative 'preference direction' for the
three features. wk_generator.py uses these directions + dataset
distribution to compute the actual numeric w_k.
"""

# Direction tokens:
#   "high"    -> hospital prefers samples with a HIGH value of that feature
#   "low"     -> hospital prefers samples with a LOW value of that feature
#   "neutral" -> no strong preference

def create_hospital_archetypes() -> list[dict]:
    archetypes = [
        {
            "id": 0,
            "name": "NorthCare_Tertiary_Melanoma_Institute",
            "type": "tertiary melanoma institute",
            "clinical_role": "High-end tertiary referral center for confirmed melanoma cases.",
            "expected_pattern": (
                "Receives biopsy/histopathology-confirmed cases, primarily dermoscopic imaging, "
                "with full Fitzpatrick spectrum but slight bias to lighter skin (referral pattern)."
            ),
            "directions": {"phi": "neutral", "tau": "high", "lambda": "high"},
            "rationale": (
                "Tertiary melanoma centers process gold-standard pathology-confirmed cases "
                "and almost always work with dermoscopic images."
            ),
        },
        {
            "id": 1,
            "name": "MetroSkin_Dermatology_Research_Clinic",
            "type": "urban dermatology research clinic",
            "clinical_role": "Urban academic dermatology research clinic with strong dermoscopy.",
            "expected_pattern": "Mostly dermoscopic, frequent expert-consensus confirmations.",
            "directions": {"phi": "neutral", "tau": "high", "lambda": "high"},
            "rationale": "Research clinics perform dermoscopy and rigorous confirmations.",
        },
        {
            "id": 2,
            "name": "CityGeneral_Multispecialty_Hospital",
            "type": "general multispecialty hospital",
            "clinical_role": "General hospital with mixed dermatology workload.",
            "expected_pattern": "Mixed image types, mixed confirmations, balanced skin types.",
            "directions": {"phi": "neutral", "tau": "neutral", "lambda": "neutral"},
            "rationale": "General hospitals see everything — should be near-balanced.",
        },
        {
            "id": 3,
            "name": "Greenfield_Rural_Community_Health_Center",
            "type": "rural community health center",
            "clinical_role": "Rural community center with limited imaging equipment.",
            "expected_pattern": (
                "Mostly clinical/overview images, follow-up or clinical-only diagnoses, "
                "and broader skin type distribution."
            ),
            "directions": {"phi": "high", "tau": "low", "lambda": "low"},
            "rationale": (
                "Rural centers serve more diverse skin types, lack dermoscopes, and "
                "often diagnose clinically without biopsy."
            ),
        },
        {
            "id": 4,
            "name": "TeleDerm_National_Screening_Network",
            "type": "teledermatology screening network",
            "clinical_role": "Remote teledermatology screening platform.",
            "expected_pattern": "Mobile/clinical images, low-tier confirmations, all skin types.",
            "directions": {"phi": "neutral", "tau": "low", "lambda": "low"},
            "rationale": "Tele-screening uses mobile/clinical photos with minimal confirmation.",
        },
        {
            "id": 5,
            "name": "Lakeside_Regional_Mixed_Hospital",
            "type": "regional mixed hospital",
            "clinical_role": "Regional hospital serving a mixed population.",
            "expected_pattern": "Balanced mix slightly skewed to clinical photos.",
            "directions": {"phi": "neutral", "tau": "neutral", "lambda": "neutral"},
            "rationale": "Regional hospitals are mixed; near-neutral preferences.",
        },
        {
            "id": 6,
            "name": "Pinewood_Oncology_Biopsy_Center",
            "type": "oncology biopsy center",
            "clinical_role": "Specialty biopsy / pathology referral center.",
            "expected_pattern": "Strongly histopathology-confirmed, often dermoscopic.",
            "directions": {"phi": "neutral", "tau": "high", "lambda": "high"},
            "rationale": "Biopsy centers see only confirmed lesions — Tier 1 confirmation dominant.",
        },
        {
            "id": 7,
            "name": "SunRay_Community_Outreach_Clinic",
            "type": "community outreach clinic",
            "clinical_role": "Outreach clinic in underserved areas.",
            "expected_pattern": "Diverse skin types (incl. darker), low-resource imaging.",
            "directions": {"phi": "high", "tau": "low", "lambda": "low"},
            "rationale": "Outreach clinics serve more darker skin types and lack pathology access.",
        },
        {
            "id": 8,
            "name": "Westbridge_Academic_Dermoscopy_Unit",
            "type": "academic dermoscopy unit",
            "clinical_role": "Academic dermoscopy training & evaluation unit.",
            "expected_pattern": "Almost exclusively dermoscopic, expert consensus heavy.",
            "directions": {"phi": "neutral", "tau": "high", "lambda": "neutral"},
            "rationale": "Academic dermoscopy units stress dermoscopy quality strongly.",
        },
        {
            "id": 9,
            "name": "GoldenSun_Public_Health_Screening_Camp",
            "type": "public health screening camp",
            "clinical_role": "Mass screening camps for early detection.",
            "expected_pattern": "Clinical/mobile photos, weak confirmations, all skin types.",
            "directions": {"phi": "high", "tau": "low", "lambda": "low"},
            "rationale": "Screening camps capture diverse populations with limited tools.",
        },
    ]
    assert len(archetypes) == 10
    return archetypes