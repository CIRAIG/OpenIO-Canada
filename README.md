**OpenIO Canada v2.0**

Class that creates Multi-Regional symmetric Environmentally Extended Input-Output (EEIO) tables for Canada. OpenIO 
operates at the provincial level. It can thus be used to compare the environmental impacts of value chains from 
Quebec and Ontario for example.

Includes 280 pollutants in 3 compartments (air, water and soil). These flows are linked to the IMPACT World+ 
life cycle impact assessment methodology.

Tables are available both in _ixi_ (industry) and _pxp_ (product) formats.
The fixed industry sales structure assumption was used to generate the _ixi_ format and the industry technology 
assumption was used for the _pxp_ format. More transformation models might be added in the future.

Unfortunately, GHG emissions are not disaggregated and were precompiled using IPCC2007 impact factors, which are 
outdated. Disaggregated GHG emissions might be made available by StatCAN in the future.
Water and primary energy flows will also be implemented in the future.

Data used:
- https://www150.statcan.gc.ca/n1/en/catalogue/15-602-X
- https://open.canada.ca/data/en/dataset/1fb7d8d4-7713-4ec6-b957-4a882a84fed3
- https://www150.statcan.gc.ca/n1/tbl/csv/38100097-eng.zip

This project was in part funded by Shared Services Canada (SSC) but they are not responsible for any data/results 
obtained from open IO Canada.