# ACTIVE TASK
Generate the complete database schema for the project data without loading full files into context.

# CONSTRAINTS
* DO NOT attempt to read the raw CSVs or shapefiles directly into your LLM context window.
* Write a temporary Python script (`temp_schema.py`) using `pandas` and `geopandas`.
* The script must read only the headers, data types, and the first 2 rows of the following representative files:
  1. `Data/ETM/Ticket-Data-2026-03-01.csv`
  2. `Public Transport/Metro Rail/Operational_Metro_Stns.shp`
  3. `Public Transport/MTC/routes_merged.shp`
* Have the script format this information cleanly and append it to `data-schema.md`.
* Execute the script. 
* Delete `temp_schema.py` once `data-schema.md` is updated.