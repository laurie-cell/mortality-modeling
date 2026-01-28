NUMERICAL_COLS = [
    "length_of_stay_hospital", # in days (float)
    "length_of_stay_unit" # in days (float)
]

CATEGORICAL_COLS = [
    "specialty",
    "last_acute_ward",
    "diagnostic_ICD",
    "diagnostic_group",
    "procedure"
]

BINARY_COLS = [
    "death_within_1_day_of_hospital_discharge", # "Yes"/"No"
    "review_flag" # "Y"/"N"
]

TIMESTAMP_COLS = [
    "hospital_admission",
    "hospital_discharge",
    "unit_admission",  # Note: actual column name in CSV is "unit _admission" (with space)
    "unit_discharge",
    "death_date"
]
# NOTE: death_date is reported as M/D/YYYY, other cols are M/D/YY H:M
