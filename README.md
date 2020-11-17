**OpenIO Canada v1.0**

Class that creates symmetric Environmentally Extended Input-Output (EEIO) tables for Canada.

Includes 280 pollutants in 3 compartments (air, water and soil). These flows are linked to the IMPACT World+ 
life cycle impact assessment methodology.

In v1.0, open IO operates at the national scale.

Tables are available both in _ixi_ (industry) and _pxp_ (product) formats.
The fixed industry sales structure assumption was used to generate the _ixi_ format and the industry technology 
assumption was used for the _pxp_ format. More transformation models might be added in the future.

Unfortunately, GHG emissions are not disaggregated and were precompiled using IPCC2007 impact factors, which are 
outdated. 

Data used:
- https://www150.statcan.gc.ca/n1/en/catalogue/15-602-X
- https://open.canada.ca/data/en/dataset/1fb7d8d4-7713-4ec6-b957-4a882a84fed3
- https://www150.statcan.gc.ca/n1/tbl/csv/38100097-eng.zip


More documentation will come with v2.0.