library(httr)
library(plyr)
library(glue)
library(tidyverse)
library(RJSONIO)


age_range <- function(age) {
## Maps an age to an appropriate decimal range
## e.g. 22 -> "[20-30)"
  low = round_any(age, 10, f=floor)
  high = low + 10
  glue("[{low}-{high})")
}

test_df <- tibble::tribble(
  ~race,  ~gender,      ~age, ~admission_type_id, ~discharge_disposition_id, ~admission_source_id, ~time_in_hospital, ~num_lab_procedures, ~num_procedures, ~num_medications, ~number_outpatient, ~number_emergency, ~number_inpatient, ~number_diagnoses, ~max_glu_serum, ~A1Cresult, ~metformin, ~repaglinide, ~nateglinide, ~chlorpropamide, ~glimepiride, ~acetohexamide, ~glipizide, ~glyburide, ~tolbutamide, ~pioglitazone, ~rosiglitazone, ~acarbose, ~miglitol, ~troglitazone, ~tolazamide, ~examide, ~citoglipton, ~insulin, ~`glyburide-metformin`, ~`glipizide-metformin`, ~`glimepiride-pioglitazone`, ~`metformin-rosiglitazone`, ~`metformin-pioglitazone`,
  "Caucasian", "Female", 22,                 2L,                        3L,                   7L,                6L,                 68L,              0L,              14L,                 0L,                0L,                0L,                9L,         "None",       ">7",       "No",         "No",         "No",            "No",         "No",           "No",       "No",       "No",         "No",          "No",           "No",      "No",      "No",          "No",        "No",     "No",         "No",     "No",                 "No",                 "No",                      "No",                     "No",                    "No"
)



get_prediction <- function(df, url="http://127.0.0.1:5000/predict") {
  request <- toJSON(as.list(df))
  response <- POST(url = url, body = request)
  
  if (response$status_code != 200) {
    stop("Somethig gone wrong!")
  }
  predictions <- unlist(content(response,as = "parsed"))
  return(predictions)
}

diag_codes <- c(
  '242 : Thyrotoxicosis with or without goiter',
  '244 : Acquired hypothyroidism',
  '245 : Thyroiditis',
  '246 : Other disorders of thyroid',
  '249 : Secondary diabetes mellitus',
  '250 : Diabetes mellitus',
  '251 : Other disorders of pancreatic internal secretion',
  '252 : Disorders of parathyroid gland',
  '252.0 : Hyperparathyroidism',
  '253 : Disorders of the pituitary gland and its hypothalamic control',
  '254 : Diseases of thymus gland',
  '255 : Disorders of adrenal glands',
  '255.1 : Hyperaldosteronism',
  '255.4 : Corticoadrenal insufficiency',
  '256 : Ovarian dysfunction',
  '256.3 : Other ovarian failure',
  '257 : Testicular dysfunction',
  '258 : Polyglandular dysfunction and related disorders',
  '258.0 : Polyglandular activity in multiple endocrine adenomatosis',
  '259 : Other endocrine disorders',
  '259.5 : Androgen insensitivity syndrome',
  '260-269 : NUTRITIONAL DEFICIENCIES ',
  '263 : Other and unspecified protein-calorie malnutrition',
  '264 : Vitamin A deficiency',
  '265 : Thiamine and niacin deficiency states',
  '266 : Deficiency of B-complex components',
  '268 : Vitamin D deficiency',
  '269 : Other nutritional deficiencies',
  '270 : Disorders of amino-acid transport and metabolism'
)
